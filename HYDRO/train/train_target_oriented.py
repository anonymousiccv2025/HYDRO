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

from HYDRO.model.arcface import iresnet100_sparse_perceptual
from HYDRO.model.threedmm import Encoder3DMM
from HYDRO.model.eye_model import FAN
from HYDRO.model.networks import Generator
from HYDRO.model.residual_discriminator import Discriminator, EyeDiscriminator
from HYDRO.loss.loss import AdversarialHingeLoss, SparsePerceptualSimilarityLoss
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

    if args.lambda_gaze > 0.0 or args.eye_discrimination:
        heatmap = FAN(4, "False", "False", 98)
        heatmap.load_state_dict(torch.load('../../pretrained/adaptive_wing/WFLW_4HG.pth')["state_dict"])
        heatmap.eval()
        heatmap.requires_grad_(False)
        heatmap.to(args.device, dtype=weight_dtype)

    generator = Generator(**cfg)
    discriminator = Discriminator()

    ifsr_criterion = SparsePerceptualSimilarityLoss().to(args.device)
    adv_criterion = AdversarialHingeLoss().to(args.device)

    arcface.requires_grad_(False)
    threedmm.requires_grad_(False)
    generator.train()
    discriminator.train()

    arcface.to(args.device, dtype=weight_dtype)
    threedmm.to(args.device)
    generator.to(args.device)
    discriminator.to(args.device)

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

    if args.eye_discrimination:
        eye_discriminator = EyeDiscriminator()
        eye_discriminator.train()
        eye_discriminator.to(args.device)

        optimizer_eye_discriminator = optimizer_class(
            eye_discriminator.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        grad_scaler_eye_discriminator = torch.cuda.amp.GradScaler()

    # Optimizer creation
    optimizer_generator = optimizer_class(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    grad_scaler_generator = torch.cuda.amp.GradScaler()

    optimizer_discriminator = optimizer_class(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    grad_scaler_discriminator = torch.cuda.amp.GradScaler()
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

        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])
        grad_scaler_discriminator.load_state_dict(checkpoint["grad_scaler_discriminator"])

        start = checkpoint["step"] + 1
    elif args.resume_training_model_index is not None:
        checkpoint = torch.load(
            args.checkpoint_dir + "/" + str(args.resume_training_model_index) + "_model_state.pth")

        generator.load_state_dict(checkpoint["generator"])
        optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        grad_scaler_generator.load_state_dict(checkpoint["grad_scaler_generator"])

        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])
        grad_scaler_discriminator.load_state_dict(checkpoint["grad_scaler_discriminator"])

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
        discriminator.requires_grad_(False)
        if args.eye_discrimination: eye_discriminator.requires_grad_(False)
        with torch.autocast(device_type='cuda', dtype=weight_dtype):
            with torch.no_grad():
                t_condition, _ = arcface(id_transform(target))
                _, t_feature_maps = arcface(id_transform(target_distorted))
                t_shape = threedmm(mm_transform(target_distorted))

                t_expression = t_shape[1]
                t_pose = t_shape[3]

                if args.lambda_gaze > 0.0 or args.eye_discrimination:
                    t_gaze_left, t_gaze_right = heatmap.detect_landmarks(target_distorted)

                if args.eye_discrimination:
                    t_left_eye, t_right_eye = heatmap.extract_eyes(target_distorted, t_gaze_left, t_gaze_right)

            raw, mask = generator(target_distorted, t_condition)
            raw_masked = raw * mask + (1 - mask) * target_distorted

            r_condition, r_feature_maps = arcface(id_transform(raw))
            rm_condition, rm_feature_maps = arcface(id_transform(raw_masked))

            r_shape = threedmm(mm_transform(raw))
            rm_shape = threedmm(mm_transform(raw_masked))

            r_expression = r_shape[1]
            r_pose = r_shape[3]

            rm_expression = rm_shape[1]
            rm_pose = rm_shape[3]

            if args.lambda_gaze > 0.0 or args.eye_discrimination:
                r_gaze_left, r_gaze_right = heatmap.detect_landmarks(raw)
                rm_gaze_left, rm_gaze_right = heatmap.detect_landmarks(raw_masked)

                if args.eye_discrimination:
                    rm_left_eye, rm_right_eye = heatmap.extract_eyes(raw_masked if step % 2 == 0 else raw,
                                                                     t_gaze_left, t_gaze_right)

                    fake_left, fake_right, g_sim, _ = eye_discriminator(rm_left_eye, rm_right_eye)

                    loss_eye_adversarial = adv_criterion(fake_left, mode='gen', dim=(1, 2, 3))
                    loss_eye_adversarial += adv_criterion(fake_right, mode='gen', dim=(1, 2, 3))
                    loss_eye_adversarial *= args.lambda_eye_adversarial

                    loss_eye_similarity = F.binary_cross_entropy_with_logits(g_sim,
                                                                             torch.ones_like(g_sim.detach(),
                                                                                             device=g_sim.device))
                    loss_eye_similarity *= args.lambda_eye_similarity

            generator_fake_prediction = discriminator(raw_masked if step % 2 == 0 else raw)
            loss_adversarial = adv_criterion(generator_fake_prediction, mode='gen', dim=(1, 2, 3))

            # Drive identity away in high level layers...
            z_0_norm = torch.nn.functional.normalize(t_condition.detach(), dim=1)
            z_1_norm = torch.nn.functional.normalize(r_condition, dim=1)

            loss_identity = torch.sum(z_0_norm * z_1_norm, dim=1) + 1

            z_0_norm = torch.nn.functional.normalize(t_condition.detach(), dim=1)
            z_1_norm = torch.nn.functional.normalize(rm_condition, dim=1)

            loss_identity += torch.sum(z_0_norm * z_1_norm, dim=1) + 1
            loss_identity *= args.lambda_identity

            # Perceptual Identity loss
            loss_ifsr = ifsr_criterion(t_feature_maps, r_feature_maps)
            loss_ifsr += ifsr_criterion(t_feature_maps, rm_feature_maps)
            loss_ifsr *= args.lambda_ifsr

            # reconstruction loss
            loss_reconstruction = torch.mean(torch.abs(raw_masked - target_distorted), dim=(1, 2, 3))
            loss_reconstruction += torch.mean(torch.abs(raw - target_distorted), dim=(1, 2, 3))
            loss_reconstruction *= args.lambda_reconstruction

            # mask loss
            loss_mask = torch.mean(torch.abs(mask), dim=(1, 2, 3))
            loss_mask *= args.lambda_mask

            # shape loss
            loss_shape = torch.mean(torch.abs(t_expression - r_expression), dim=1)
            loss_shape += torch.mean(torch.abs(t_expression - rm_expression), dim=1)
            loss_shape += torch.mean(torch.abs(t_pose - r_pose), dim=1)
            loss_shape += torch.mean(torch.abs(t_pose - rm_pose), dim=1)
            loss_shape *= args.lambda_shape

            # gaze loss
            loss_gaze = 0.0
            if args.lambda_gaze > 0.0:
                loss_gaze = torch.mean(torch.abs(t_gaze_left - r_gaze_left), dim=(1, 2))
                loss_gaze += torch.mean(torch.abs(t_gaze_right - rm_gaze_right), dim=(1, 2))
                loss_gaze += torch.mean(torch.abs(t_gaze_left - rm_gaze_left), dim=(1, 2))
                loss_gaze += torch.mean(torch.abs(t_gaze_right - rm_gaze_right), dim=(1, 2))
                loss_gaze *= args.lambda_gaze

            loss_generator = (loss_adversarial + loss_identity +
                              loss_reconstruction + loss_ifsr +
                              loss_mask + loss_shape + loss_gaze)

            if args.eye_discrimination:
                loss_generator += loss_eye_similarity + loss_eye_adversarial

            loss_generator = loss_generator.mean()

        # 2.5 Optimize model
        optimizer_generator.zero_grad(set_to_none=True)
        grad_scaler_generator.scale(loss_generator).backward()
        grad_scaler_generator.unscale_(optimizer_generator)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        grad_scaler_generator.step(optimizer_generator)
        grad_scaler_generator.update()

        with (torch.autocast(device_type='cuda', dtype=weight_dtype)):
            target_distorted = target_distorted.detach()
            discriminator.requires_grad_(True)
            target_distorted.requires_grad_(True)
            # ======================================================================================================
            # ======================================== Train Discriminator =========================================
            # ======================================================================================================
            discriminator_real_prediction = discriminator(target_distorted)
            discriminator_fake_prediction = discriminator(raw_masked.detach())  # Change this?

            loss_fake_prediction = adv_criterion(discriminator_fake_prediction, real=False, mode='dis')
            loss_real_prediction = adv_criterion(discriminator_real_prediction, real=True, mode='dis')
            epsilon_penalty = discriminator_real_prediction.pow(2).mean(dim=(1, 2, 3)) * 0.001

            # gradient penalty
            grad = torch.autograd.grad(
                outputs=grad_scaler_discriminator.scale(discriminator_real_prediction),
                inputs=target_distorted,
                grad_outputs=torch.ones_like(discriminator_real_prediction),
                create_graph=True,
                only_inputs=True,
            )[0]
            inv_scale = 1.0 / (grad_scaler_discriminator.get_scale() + 1e-6)
            grad = grad * inv_scale
            grad = grad.square().sum(dim=[1, 2, 3])
            lambd_ = 0.005
            gradient_penalty = grad * lambd_

            loss_discriminator = loss_fake_prediction + loss_real_prediction + gradient_penalty + epsilon_penalty
            loss_discriminator = loss_discriminator.mean()

        optimizer_discriminator.zero_grad(set_to_none=True)
        grad_scaler_discriminator.scale(loss_discriminator).backward()
        grad_scaler_discriminator.unscale_(optimizer_discriminator)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        grad_scaler_discriminator.step(optimizer_discriminator)
        grad_scaler_discriminator.update()

        if args.eye_discrimination:
            eye_discriminator.requires_grad_(True)
            t_left_eye.requires_grad_(True)
            t_right_eye.requires_grad_(True)
            with (torch.autocast(device_type='cuda', dtype=weight_dtype)):
                d_fake_left, d_fake_right, _, _ = eye_discriminator(rm_left_eye.detach(), rm_right_eye.detach())
                d_real_left, d_real_right, d_sim_same, d_sim_diff = eye_discriminator(t_left_eye, t_right_eye)

                loss_eye_adversarial_d = adv_criterion(d_fake_left, real=False, mode='dis')
                loss_eye_adversarial_d += adv_criterion(d_fake_right, real=False, mode='dis')

                loss_eye_adversarial_d += adv_criterion(d_real_left, real=True, mode='dis')
                loss_eye_adversarial_d += adv_criterion(d_real_right, real=True, mode='dis')

                loss_eye_adversarial_d /= 4

                loss_eye_similarity_d = F.binary_cross_entropy_with_logits(d_sim_same,
                                                                         torch.ones_like(d_sim_same.detach(),
                                                                                         device=d_sim_same.device))
                loss_eye_similarity_d += F.binary_cross_entropy_with_logits(d_sim_diff,
                                                                          torch.zeros_like(d_sim_diff.detach(),
                                                                                           device=d_sim_diff.device))
                loss_eye_similarity_d *= args.lambda_eye_similarity * 2

                epsilon_eye_penalty = (d_real_left.pow(2).mean(dim=(1, 2, 3)) +
                                       d_real_right.pow(2).mean(dim=(1, 2, 3))) * 0.0005

                # gradient penalty
                lambd_ = 0.005
                inv_scale = 1.0 / (grad_scaler_eye_discriminator.get_scale() + 1e-6)
                grad_left = torch.autograd.grad(
                    outputs=[grad_scaler_eye_discriminator.scale(d_real_left), grad_scaler_eye_discriminator.scale(d_real_right)],
                    inputs=[t_left_eye, t_right_eye],
                    grad_outputs=[torch.ones_like(d_real_left), torch.ones_like(d_real_left)],
                    create_graph=True,
                    only_inputs=True,
                )[0]
                grad_left = grad_left * inv_scale
                grad_left = grad_left.square().sum(dim=[1, 2, 3])

                gradient_eye_penalty = grad_left * lambd_
                loss_eye_discriminator = (loss_eye_adversarial_d + loss_eye_similarity_d +
                                          epsilon_eye_penalty + gradient_eye_penalty)
                loss_eye_discriminator = loss_eye_discriminator.mean()

            optimizer_eye_discriminator.zero_grad(set_to_none=True)
            grad_scaler_eye_discriminator.scale(loss_eye_discriminator).backward()
            grad_scaler_eye_discriminator.unscale_(optimizer_eye_discriminator)
            torch.nn.utils.clip_grad_norm_(eye_discriminator.parameters(), max_norm=1.0)
            grad_scaler_eye_discriminator.step(optimizer_eye_discriminator)
            grad_scaler_eye_discriminator.update()

        # ======================================================================================================
        # ============================================= Logging ================================================
        # ======================================================================================================
        logging_dictionary_losses = {}
        logging_dictionary_images = {}
        if step % args.log_every == 0:
            logging_dictionary_losses['generator/loss_generator'] = loss_generator.mean().detach().item()
            logging_dictionary_losses['generator/loss_identity'] = loss_identity.mean().detach().item()
            logging_dictionary_losses['generator/loss_reconstruction'] = loss_reconstruction.mean().detach().item()
            logging_dictionary_losses['generator/loss_shape'] = loss_shape.mean().detach().item()
            logging_dictionary_losses['generator/loss_ifsr'] = loss_ifsr.mean().detach().item()
            logging_dictionary_losses['generator/loss_adversarial'] = loss_adversarial.mean().detach().item()
            if args.lambda_gaze > 0.0:
                logging_dictionary_losses['generator/loss_gaze'] = loss_gaze.mean().detach().item()

            logging_dictionary_losses[
                'discriminator/loss_fake_prediction'] = loss_fake_prediction.mean().detach().item()
            logging_dictionary_losses[
                'discriminator/loss_real_prediction'] = loss_real_prediction.mean().detach().item()
            logging_dictionary_losses['discriminator/epsilon_penalty'] = epsilon_penalty.mean().detach().item()
            logging_dictionary_losses['discriminator/gradient_penalty'] = gradient_penalty.mean().detach().item()

            if args.eye_discrimination > 0.0:
                logging_dictionary_losses[
                    'generator/loss_eye_similarity'] = loss_eye_similarity.mean().detach().item()
                logging_dictionary_losses[
                    'generator/loss_eye_adversarial'] = loss_eye_adversarial.mean().detach().item()

                logging_dictionary_losses[
                    'eye_discriminator/loss_eye_adversarial_d'] = loss_eye_adversarial_d.mean().detach().item()
                logging_dictionary_losses[
                    'eye_discriminator/loss_eye_similarity_d'] = loss_eye_similarity_d.mean().detach().item()
                logging_dictionary_losses[
                    'eye_discriminator/epsilon_penalty'] = epsilon_eye_penalty.mean().detach().item()
                logging_dictionary_losses[
                    'eye_discriminator/gradient_penalty'] = gradient_eye_penalty.mean().detach().item()
            log_losses(logger, logging_dictionary_losses, step)

        # ======================================================================================================
        # ========================================== Visualization =============================================
        # ======================================================================================================
        if step % args.sample_every == 0:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=weight_dtype):
                logging_dictionary_images['generator/target'] = target_distorted * 0.5 + 0.5
                logging_dictionary_images['generator/deid'] = raw * 0.5 + 0.5
                logging_dictionary_images['generator/deid_masked'] = raw_masked * 0.5 + 0.5
                logging_dictionary_images['generator/mask'] = mask
                log_images(logger, logging_dictionary_images, step)

        if step % args.save_every == 0:
            model_index = int(np.floor(step / args.checkpoint_every)) * args.checkpoint_every
            torch.save({"generator": generator.state_dict(),
                        "optimizer_generator": optimizer_generator.state_dict(),
                        "grad_scaler_generator": grad_scaler_generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer_discriminator": optimizer_discriminator.state_dict(),
                        "grad_scaler_discriminator": grad_scaler_discriminator.state_dict(),

                        "eye_discriminator":
                            eye_discriminator.state_dict() if args.eye_discrimination else None,
                        "optimizer_eye_discriminator":
                            optimizer_eye_discriminator.state_dict() if args.eye_discrimination else None,
                        "grad_scaler_eye_discriminator":
                            grad_scaler_eye_discriminator.state_dict() if args.eye_discrimination else None,

                        "step": step},
                       args.checkpoint_dir + "/" + str(model_index) + "_model_state.pth")

        progress_bar.update(1)
        progress_bar.set_postfix(**{"loss": loss_generator.detach().item()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    LOG_NAME = 'HYDRO_target_oriented'

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

    parser.add_argument('--model_config', type=str, default="baseline_plus_addition_skips_attention")

    # Intervals: Checkpointing, sampling images, logging loss
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--checkpoint_every', type=int, default=25000)

    # Data specifics
    parser.add_argument('--train_dir', type=str, default="path/to/train/data/")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_type', type=str, default="float16")

    # Loss weights
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--lambda_reconstruction', type=float, default=0.5)
    parser.add_argument('--lambda_mask', type=float, default=3e-3)
    parser.add_argument('--lambda_shape', type=float, default=1.0)
    parser.add_argument('--lambda_ifsr', type=float, default=1.0)
    parser.add_argument('--lambda_gaze', type=float, default=25)
    parser.add_argument('--lambda_eye_similarity', type=float, default=0)
    parser.add_argument('--lambda_eye_adversarial', type=float, default=0.5)

    parser.add_argument('--eye_discrimination', type=bool, default=True)

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

    args = parser.parse_args()

    train(args)
