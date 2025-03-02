import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import torch.nn.functional as F
import torchvision
import os
import tqdm
import numpy as np
import json
import albumentations as A
import HYDRO.config.config as config_module
import lpips

from datetime import datetime

from HYDRO.model.arcface import iresnet100_sparse_perceptual
from HYDRO.model.networks import Generator, DiffusionGenerator
from HYDRO.model.tracker import TrackerTorch
from diffusers import DPMSolverMultistepScheduler
from HYDRO.data.dataset import DeIDDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from albumentations.pytorch import ToTensorV2

from torchvision import transforms as TF


def id_transform(x, h=32, w=32, resize=112):
    return F.interpolate(x[:, :, h:-h, w:-w], [resize, resize])


def mm_transform(x, resize=224):
    return F.interpolate(x * 0.5 + 0.5, [resize, resize])


def log_images(writer, img_dict, step):
    for key in img_dict.keys():
        grid = torch.clip(torchvision.utils.make_grid(img_dict[key].detach(), nrow=4), 0, 1)
        writer.add_image(key, grid, global_step=step)


def log_losses(logger, loss_dict, step):
    for key in loss_dict.keys():
        logger.add_scalar(key, loss_dict[key], global_step=step)


def get_timesteps(num_inference_steps, strength, scheduler):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * scheduler.order)

    return timesteps, num_inference_steps - t_start


def log_validation(target, t_condition, generator, weight_dtype, steps=30, strength=0.8):
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=weight_dtype):
            scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000,
                                                    beta_start=0.00085,
                                                    beta_end=0.012,
                                                    beta_schedule="scaled_linear",
                                                    trained_betas=None,
                                                    variance_type="fixed_small",
                                                    prediction_type="epsilon",
                                                    thresholding=False,
                                                    dynamic_thresholding_ratio=0.995,
                                                    sample_max_value=1.0,
                                                    timestep_spacing="leading",
                                                    steps_offset=1,
                                                    rescale_betas_zero_snr=False)
            scheduler.set_timesteps(steps, device=target.device)

            timesteps, num_inference_steps = get_timesteps(
                num_inference_steps=steps, strength=strength, scheduler=scheduler
            )

            latent_timestep = timesteps[:1].repeat(target.shape[0] * 1)

            noise = torch.randn(target.shape, device=target.device)

            noisy_target = scheduler.add_noise(target, noise, latent_timestep)

            for i, t in enumerate(timesteps):
                timestep_batch = t.repeat(noisy_target.shape[0])
                noisy_target_input = scheduler.scale_model_input(noisy_target, t)

                noise_prediction = generator(noisy_target_input,
                                             t_condition, timestep_batch)

                noisy_target = scheduler.step(noise_prediction, t, noisy_target)
                noisy_target = noisy_target.prev_sample

            validation_images = noisy_target

        return validation_images


def train(args):
    # Networks
    cfg = getattr(config_module, args.generator_config)
    cfg_rec = getattr(config_module, args.reconstructor_config)
    cfg_diff = getattr(config_module, args.diffusion_config)

    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.data_type == 'bfloat16':
        weight_dtype = torch.bfloat16
    elif args.data_type == 'float16':
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    arcface = iresnet100_sparse_perceptual()
    arcface.load_state_dict(torch.load(args.arcface_path))
    arcface.eval()

    generator = Generator(**cfg)
    generator.load_state_dict(torch.load(args.generator_path)["generator"])
    generator.eval()

    diffusion = DiffusionGenerator(**cfg_diff)
    diffusion.load_state_dict(torch.load(args.diffusion_path)["generator"])
    diffusion.eval()

    itm = TrackerTorch(track=False)

    reconstructor = Generator(**cfg_rec)

    loss_fn_vgg = lpips.LPIPS(net='vgg', lpips=False).to(args.device)

    arcface.requires_grad_(False)
    generator.requires_grad_(False)
    diffusion.requires_grad_(False)
    itm.requires_grad_(False)

    arcface.to(args.device, dtype=weight_dtype)
    generator.to(args.device, dtype=weight_dtype)
    itm.to(args.device, dtype=weight_dtype)
    if args.apply_diffusion:
        diffusion.to(args.device)
    reconstructor.to(args.device)

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer_reconstructor = optimizer_class(
        reconstructor.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    grad_scaler_reconstructor = torch.cuda.amp.GradScaler()

    # Data
    transforms = A.Compose([A.Resize(args.image_size, args.image_size),
                            A.HorizontalFlip(p=0.5),
                            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ToTensorV2(), ])

    distort = torch.nn.Sequential(TF.RandomRotation(10),
                                  TF.RandomResizedCrop(size=args.image_size, scale=(0.9, 1.0)))

    train_dataset = DeIDDataset(root_dir=args.train_dir, transform=transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=args.num_workers == 0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Logging
    if args.make_directories:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.config_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger = SummaryWriter(log_dir=args.log_dir)

    total_config = args.__dict__

    # Loading
    if args.resume_training_model_index is not None and args.resume_path is not None:
        checkpoint = torch.load(
            args.resume_path + "/" + str(args.resume_training_model_index) + "_model_state.pth")

        generator.load_state_dict(checkpoint["generator"])
        optimizer_reconstructor.load_state_dict(checkpoint["optimizer_generator"])
        grad_scaler_reconstructor.load_state_dict(checkpoint["grad_scaler_generator"])

        start = checkpoint["step"] + 1
    elif args.resume_training_model_index is not None:
        checkpoint = torch.load(
            args.checkpoint_dir + "/" + str(args.resume_training_model_index) + "_model_state.pth")

        generator.load_state_dict(checkpoint["generator"])
        optimizer_reconstructor.load_state_dict(checkpoint["optimizer_generator"])
        grad_scaler_reconstructor.load_state_dict(checkpoint["grad_scaler_generator"])

        start = checkpoint["step"] + 1
    else:
        start = 0

    # Dump settings to text file
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')

    if not args.resume_training_model_index:
        config_path = args.config_dir + '/options_' + date + '.txt'
    else:
        config_path = args.config_dir + '/options_' + date + '_resume.txt'

    with open(config_path, 'w') as f:
        json.dump(total_config, f, indent=2)

    progress_bar = tqdm.tqdm(
        range(len(train_dataset) // args.batch_size * args.epochs),
        initial=start,
        desc="Steps",
    )

    train_iterator = iter(train_dataloader)
    if args.fit_on_single_batch:
        print("WARNING!!! The script is running with train on single batch, "
              "and should only be used for debugging purposes!")
        target = next(train_iterator)
    # Training loop
    for step in range(start, len(train_dataset)):

        if not args.fit_on_single_batch:
            target = next(train_iterator)

        target = target.to(args.device)

        target_distorted = distort(target)
        with torch.autocast(device_type='cuda', dtype=weight_dtype):
            with torch.no_grad():
                t_condition, t_feature_maps = arcface(id_transform(target))

                B, _ = t_condition.shape

                margin = torch.rand(B, 1, device=args.device) * 0.3 if args.random_margin else None

                itm_condition = itm(t_condition, margin)

                t_norm = torch.nn.functional.normalize(t_condition, dim=1)
                itm_norm = torch.nn.functional.normalize(itm_condition, dim=1)

                cos_d_t_to_itm = (1 - torch.sum(t_norm * itm_norm, dim=1)).mean()
                cos_d_t_to_itm_min = (1 - torch.sum(t_norm * itm_norm, dim=1)).min()
                cos_d_t_to_itm_max = (1 - torch.sum(t_norm * itm_norm, dim=1)).max()

                raw, mask = generator(target_distorted, itm_condition)
                deid = raw * mask + (1 - mask) * target_distorted

                deid_unchanged = deid

                if args.apply_noise or args.apply_diffusion:

                    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000,
                                                            beta_start=0.00085,
                                                            beta_end=0.012,
                                                            beta_schedule="scaled_linear",
                                                            trained_betas=None,
                                                            variance_type="fixed_small",
                                                            prediction_type="epsilon",
                                                            thresholding=False,
                                                            dynamic_thresholding_ratio=0.995,
                                                            sample_max_value=1.0,
                                                            timestep_spacing="leading",
                                                            steps_offset=1,
                                                            rescale_betas_zero_snr=False)

                    scheduler.set_timesteps(args.diffusion_inference_steps, device=args.device)

                    timesteps, num_inference_steps = get_timesteps(
                        num_inference_steps=args.diffusion_inference_steps,
                        strength=args.diffusion_strength,
                        scheduler=scheduler
                    )

                    latent_timestep = timesteps[:1].repeat(args.batch_size)

                    noise = torch.randn(deid.shape, device=deid.device)

                    noisy_deid = scheduler.add_noise(deid, noise, latent_timestep)

                    if args.apply_diffusion:
                        for i, t in enumerate(timesteps):
                            timestep_batch = t.repeat(noisy_deid.shape[0])
                            noisy_deid_input = scheduler.scale_model_input(noisy_deid, t)

                            noise_prediction = diffusion(noisy_deid_input,
                                                         -1 * itm_condition, timestep_batch)

                            noisy_deid = scheduler.step(noise_prediction, t, noisy_deid)
                            noisy_deid = noisy_deid.prev_sample

                    deid = noisy_deid
                    d_condition, _ = arcface(id_transform(deid))

                    d_norm = torch.nn.functional.normalize(d_condition, dim=1)

                    cos_d_t_to_d = (1 - torch.sum(t_norm * d_norm, dim=1)).mean()
                    cos_d_t_to_d_min = (1 - torch.sum(t_norm * d_norm, dim=1)).min()
                    cos_d_t_to_d_max = (1 - torch.sum(t_norm * d_norm, dim=1)).max()

            reconstruction = reconstructor(deid)

            r_condition, _ = arcface(id_transform(reconstruction))

            # identity loss
            r_norm = torch.nn.functional.normalize(r_condition, dim=1)

            loss_identity = 1 - torch.sum(t_norm * r_norm, dim=1)
            loss_identity *= args.lambda_identity

            # reconstruction loss
            loss_reconstruction = torch.mean(torch.abs(reconstruction - target_distorted), dim=(1, 2, 3))
            loss_reconstruction *= args.lambda_reconstruction

            # perceptual loss
            loss_perceptual = loss_fn_vgg(target_distorted, reconstruction).mean()
            loss_perceptual *= args.lambda_perceptual

            loss_reconstructor = (loss_identity + loss_reconstruction + loss_perceptual)

            loss_reconstructor = loss_reconstructor.mean()

        # 2.5 Optimize model
        optimizer_reconstructor.zero_grad(set_to_none=True)
        grad_scaler_reconstructor.scale(loss_reconstructor).backward()
        grad_scaler_reconstructor.unscale_(optimizer_reconstructor)
        torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=1.0)
        grad_scaler_reconstructor.step(optimizer_reconstructor)
        grad_scaler_reconstructor.update()

        # ======================================================================================================
        # ============================================= Logging ================================================
        # ======================================================================================================
        logging_dictionary_losses = {}
        logging_dictionary_images = {}
        if step % args.log_every == 0:
            logging_dictionary_losses['reconstructor/loss_reconstructor'] = loss_reconstructor.mean().detach().item()
            logging_dictionary_losses['reconstructor/loss_identity'] = loss_identity.mean().detach().item()
            logging_dictionary_losses['reconstructor/loss_reconstruction'] = loss_reconstruction.mean().detach().item()
            logging_dictionary_losses['reconstructor/loss_perceptual'] = loss_perceptual.mean().detach().item()

            logging_dictionary_losses['utility/cos_d_t_to_itm'] = cos_d_t_to_itm.item()
            logging_dictionary_losses['utility/cos_d_t_to_d'] = cos_d_t_to_d.item()

            logging_dictionary_losses['utility/cos_d_t_to_itm_min'] = cos_d_t_to_itm_min.item()
            logging_dictionary_losses['utility/cos_d_t_to_itm_max'] = cos_d_t_to_itm_max.item()

            logging_dictionary_losses['utility/cos_d_t_to_d_min'] = cos_d_t_to_d_min.item()
            logging_dictionary_losses['utility/cos_d_t_to_d_max'] = cos_d_t_to_d_max.item()
            log_losses(logger, logging_dictionary_losses, step)

        # ======================================================================================================
        # ========================================== Visualization =============================================
        # ======================================================================================================
        if step % args.sample_every == 0:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=weight_dtype):
                logging_dictionary_images['generator/target'] = target_distorted * 0.5 + 0.5
                logging_dictionary_images['generator/deid'] = deid * 0.5 + 0.5
                logging_dictionary_images['generator/deid_unchanged'] = deid_unchanged * 0.5 + 0.5
                logging_dictionary_images['generator/reconstruction'] = reconstruction * 0.5 + 0.5
                log_images(logger, logging_dictionary_images, step)

        if step % args.save_every == 0:
            model_index = int(np.floor(step / args.checkpoint_every)) * args.checkpoint_every
            torch.save({"reconstructor": reconstructor.state_dict(),
                        "optimizer_reconstructor": optimizer_reconstructor.state_dict(),
                        "grad_scaler_reconstructor": grad_scaler_reconstructor.state_dict(),

                        "step": step},
                       args.checkpoint_dir + "/" + str(model_index) + "_model_state.pth")

        progress_bar.update(1)
        progress_bar.set_postfix(**{"loss": loss_reconstructor.detach().item()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    LOG_NAME = 'HYDRO_reconstruction_attack'

    # Base model to target
    parser.add_argument(
        "--arcface_path",
        type=str,
        default="../pretrained/arcface/backbone.pth",
    )
    parser.add_argument(
        "--generator_path",
        type=str,
        default="../pretrained/HYDRO_target_oriented/50000_model_state.pth",
    )
    parser.add_argument(
        "--diffusion_path",
        type=str,
        default="../pretrained/HYDRO_diffusion_model/150000_model_state.pth",
    )

    parser.add_argument('--reconstructor_config', type=str, default="reconstructor_plus_addition_skips_attention")
    parser.add_argument('--generator_config', type=str, default="baseline_plus_addition_skips_attention")
    parser.add_argument('--diffusion_config', type=str, default="diffusion_plus_addition_skips_attention_mlp")

    # Intervals: Checkpointing, sampling images, logging loss
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--checkpoint_every', type=int, default=25000)

    # Data specifics
    parser.add_argument('--train_dir', type=str, default="E:/Dataset/vggface2/train_aligned_v3/data/")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_type', type=str, default="float16")

    # Loss weights
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--lambda_reconstruction', type=float, default=5.0)
    parser.add_argument('--lambda_perceptual', type=float, default=1.0)

    # General
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--adam_beta1', type=float, default=0.5)
    parser.add_argument('--adam_beta2', type=float, default=0.99)
    parser.add_argument("--adam_weight_decay", type=float, default=2e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--device', type=str, default='cuda:0')

    # Logging
    parser.add_argument('--checkpoint_dir', type=str, default=f"../../results/checkpoints/{LOG_NAME}")
    parser.add_argument('--log_dir', type=str, default=f"../../results/logs/{LOG_NAME}")
    parser.add_argument('--config_dir', type=str, default=f"../../results/configs/{LOG_NAME}")
    parser.add_argument('--make_directories', type=bool, default=True)

    # Loading / Resuming
    parser.add_argument('--resume_training_model_index', type=int, default=None)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--from_config', type=bool, default=False)  # This will override the arguments
    parser.add_argument('--from_config_path', type=str, default="")

    # Model details
    parser.add_argument('--image_size', type=int, default=256)

    # Debugging
    parser.add_argument('--fit_on_single_batch', type=bool, default=False)

    # Optimization
    parser.add_argument('--use_8bit_adam', type=bool, default=False)

    # Diffusion / Noising parameters
    parser.add_argument('--apply_noise', type=bool, default=True)
    parser.add_argument('--apply_diffusion', type=bool, default=True)
    parser.add_argument('--diffusion_strength', type=float, default=0.1)
    parser.add_argument('--diffusion_inference_steps', type=int, default=10)
    parser.add_argument('--random_margin', type=bool, default=True)

    args = parser.parse_args()

    train(args)
