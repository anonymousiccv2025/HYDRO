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

from datetime import datetime

from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

from HYDRO.model.arcface import iresnet100_sparse_perceptual
from HYDRO.model.threedmm import Encoder3DMM
from HYDRO.model.networks import DiffusionGenerator as Generator
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


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


def get_timesteps(num_inference_steps, strength, scheduler):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * scheduler.order)

    return timesteps, num_inference_steps - t_start


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
    521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
    # 521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


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
    cfg = getattr(config_module, args.model_config)

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

    threedmm = Encoder3DMM()
    threedmm.load_state_dict(torch.load(args.threedmm_path, map_location='cpu')['net_recon'])
    threedmm.eval()

    generator = Generator(**cfg)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="linear",
                                    trained_betas=None,
                                    variance_type="fixed_small",
                                    clip_sample=False,
                                    prediction_type="epsilon",
                                    thresholding=False,
                                    dynamic_thresholding_ratio=0.995,
                                    clip_sample_range=1.0,
                                    sample_max_value=1.0,
                                    timestep_spacing="leading",
                                    steps_offset=1,
                                    rescale_betas_zero_snr=False)
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(args.device)

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    alpha_schedule = alpha_schedule.to(args.device)
    sigma_schedule = sigma_schedule.to(args.device)

    arcface.requires_grad_(False)
    threedmm.requires_grad_(False)
    generator.train()

    arcface.to(args.device, dtype=weight_dtype)
    threedmm.to(args.device, dtype=weight_dtype)
    generator.to(args.device)

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
    optimizer_generator = optimizer_class(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    grad_scaler_generator = torch.cuda.amp.GradScaler()

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
        optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        grad_scaler_generator.load_state_dict(checkpoint["grad_scaler_generator"])

        start = checkpoint["step"] + 1
    elif args.resume_training_model_index is not None:
        checkpoint = torch.load(
            args.checkpoint_dir + "/" + str(args.resume_training_model_index) + "_model_state.pth")

        generator.load_state_dict(checkpoint["generator"])
        optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        grad_scaler_generator.load_state_dict(checkpoint["grad_scaler_generator"])

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

        target = distort(target)
        with torch.autocast(device_type='cuda', dtype=weight_dtype):
            with torch.no_grad():
                t_condition, t_feature_maps = arcface(id_transform(target))

                #target_cond = gaussian_blur(F.interpolate(F.interpolate(target, scale_factor=1/8), scale_factor=8))

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn((target.shape[0], target.shape[1], 1, 1),
                                                             device=target.device)

                bsz = target.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, args.timesteps, (bsz,),
                                          device=target.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target = noise_scheduler.add_noise(target, noise, timesteps)

            predicted_noise = generator(noisy_target, t_condition, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                gt_noise = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                gt_noise = noise_scheduler.get_velocity(target, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.snr_gamma is None:
                loss_generator = F.mse_loss(predicted_noise.float(), gt_noise.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss_generator = F.mse_loss(predicted_noise.float(), gt_noise.float(), reduction="none")
                loss_generator = loss_generator.mean(dim=list(range(1, len(loss_generator.shape)))) * mse_loss_weights
                loss_generator = loss_generator.mean()

            x_0 = get_predicted_original_sample(predicted_noise, timesteps, noisy_target,
                                                "epsilon", alpha_schedule, sigma_schedule)
            time_weight = torch.sigmoid(12 * (1 - timesteps / 1000) - 6)

            if args.lambda_identity > 0.0:
                z_0, x_0_feature_maps = arcface(id_transform(x_0))

                z_0_norm = torch.nn.functional.normalize(z_0, dim=1)
                t_norm = torch.nn.functional.normalize(t_condition.detach(), dim=1)

                loss_identity = ((1 - torch.sum(z_0_norm * t_norm, dim=1)) * time_weight).mean()
                loss_generator += loss_identity * args.lambda_identity

            if args.lambda_shape > 0.0:
                target_shape = threedmm(mm_transform(target))
                x_0_shape = threedmm(mm_transform(x_0))

                loss_shape = torch.mean(torch.abs(target_shape[1] - x_0_shape[1]), dim=1)
                loss_shape += torch.mean(torch.abs(target_shape[3] - x_0_shape[3]), dim=1)
                loss_shape = (loss_shape * time_weight).mean()

                loss_generator += loss_shape * args.lambda_shape

        # 2.5 Optimize model
        optimizer_generator.zero_grad()
        grad_scaler_generator.scale(loss_generator).backward()
        grad_scaler_generator.unscale_(optimizer_generator)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        grad_scaler_generator.step(optimizer_generator)
        grad_scaler_generator.update()

        # ======================================================================================================
        # ============================================= Logging ================================================
        # ======================================================================================================
        logging_dictionary_losses = {}
        logging_dictionary_images = {}
        if step % args.log_every == 0:
            logging_dictionary_losses['generator/loss_generator'] = loss_generator.mean().detach().item()
            if args.lambda_identity > 0.0:
                logging_dictionary_losses['generator/loss_identity'] = loss_identity.mean().detach().item()
            if args.lambda_shape > 0.0:
                logging_dictionary_losses['generator/loss_shape'] = loss_shape.mean().detach().item()
            log_losses(logger, logging_dictionary_losses, step)

        # ======================================================================================================
        # ========================================== Visualization =============================================
        # ======================================================================================================
        if step % args.sample_every == 0:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=weight_dtype):
                validation_image_0 = log_validation(target, t_condition, generator, weight_dtype, strength=1.0)
                validation_image_1 = log_validation(target, t_condition, generator, weight_dtype, strength=0.8)
                validation_image_2 = log_validation(target, t_condition, generator, weight_dtype, strength=0.5)
                validation_image_3 = log_validation(target, t_condition, generator, weight_dtype, strength=0.1)
                logging_dictionary_images['generator/target'] = target * 0.5 + 0.5
                logging_dictionary_images['generator/validation_image_0'] = validation_image_0 * 0.5 + 0.5
                logging_dictionary_images['generator/validation_image_1'] = validation_image_1 * 0.5 + 0.5
                logging_dictionary_images['generator/validation_image_2'] = validation_image_2 * 0.5 + 0.5
                logging_dictionary_images['generator/validation_image_3'] = validation_image_3 * 0.5 + 0.5
                log_images(logger, logging_dictionary_images, step)

        if step % args.save_every == 0:
            model_index = int(np.floor(step / args.checkpoint_every)) * args.checkpoint_every
            torch.save({"generator": generator.state_dict(),
                        "optimizer_generator": optimizer_generator.state_dict(),
                        "grad_scaler_generator": grad_scaler_generator.state_dict(),
                        "step": step},
                       args.checkpoint_dir + "/" + str(model_index) + "_model_state.pth")

        progress_bar.update(1)
        progress_bar.set_postfix(**{"loss": loss_generator.detach().item()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Example log name, used for default values if ran without arguments
    LOG_NAME = 'HYDRO_diffusion_model'

    # Base model to target
    parser.add_argument(
        "--arcface_path",
        type=str,
        default="../pretrained/arcface/backbone.pth",
    )
    parser.add_argument(
        "--threedmm_path",
        type=str,
        default="../pretrained/model_3dmm/epoch_20.pth",
    )

    parser.add_argument('--model_config', type=str, default="diffusion_plus_addition_skips_attention")

    # Intervals: Checkpointing, sampling images, logging loss
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--checkpoint_every', type=int, default=20000)

    # Data specifics
    parser.add_argument('--train_dir', type=str, default="path/to/train/data/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_type', type=str, default="float32")

    # Loss weights
    parser.add_argument('--noise_offset', type=float, default=0.05)
    parser.add_argument('--input_perturbation', type=float, default=None)
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument('--lambda_identity', type=float, default=0.0)
    parser.add_argument('--lambda_shape', type=float, default=0.0)
    parser.add_argument('--timesteps', type=str, default=200)

    # General
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.95)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay to use.")
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

    args = parser.parse_args()

    train(args)
