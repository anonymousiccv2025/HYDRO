def process_data(args, tensor_queue, done_value):
    import torch
    import numpy as np
    import json
    import torchvision

    from HYDRO.model.networks import Generator, DiffusionGenerator
    from HYDRO.model.arcface import iresnet100
    from HYDRO.model.tracker import TrackerTorch
    from HYDRO.model.threedmm import Encoder3DMM
    from HYDRO.data.dataset import GenerateEvalDataset
    from HYDRO.model.simplified_dpm_solver import SingleStepDPMSolverScheduler
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
    from HYDRO.model.eye_model import FAN
    from HYDRO.evaluation.metrics import MetricCollector, l2_error
    from HYDRO.model.hopenet import Hopenet

    import torch.nn.functional as F
    import HYDRO.config.config as config_module

    from tqdm import tqdm

    def id_transform(x, h=32, w=32, resize=112):
        return F.interpolate(x[:, :, h:-h, w:-w], [resize, resize])

    def mm_transform(x, resize=224):
        return F.interpolate(x * 0.5 + 0.5, [resize, resize])

    def hope_transform(x, resize=224):
        x = TF.center_crop(x, 224)
        x = TF.normalize(x * 0.5 + 0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return F.interpolate(x, [resize, resize])

    def gaze_transform(x, resize=256):
        x = TF.center_crop(x, 224)
        return F.interpolate(x, [resize, resize])

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

    generator = Generator(**cfg)
    generator.load_state_dict(torch.load(args.generator_path)["generator"])
    generator.eval()

    arcface = iresnet100(pretrained=False)
    arcface.load_state_dict(torch.load(args.arcface_path))
    arcface.eval()

    diffusion = DiffusionGenerator(**cfg_diff)
    diffusion.load_state_dict(torch.load(args.diffusion_path)["generator"])
    diffusion.eval()

    reconstructor = Generator(**cfg_rec)
    reconstructor.load_state_dict(torch.load(args.reconstructor_path)["reconstructor"])
    reconstructor.eval()

    heatmap = FAN(4, "False", "False", 98)
    heatmap.load_state_dict(torch.load(args.awing_path)["state_dict"])
    heatmap.eval()

    threedmm = Encoder3DMM()
    threedmm.load_state_dict(torch.load(args.threedmm_path, map_location='cpu')['net_recon'])
    threedmm.eval()

    hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    hopenet.load_state_dict(torch.load(args.hopenet_path, map_location=torch.device('cpu')))
    hopenet.eval()

    idx_tensor = torch.FloatTensor([idx for idx in range(66)])
    idx_tensor = idx_tensor.to(args.device, dtype=weight_dtype)
    softmax = torch.nn.Softmax(dim=1)

    face_tracker = TrackerTorch(track=args.track, threshold=args.threshold, margin=args.margin)
    face_tracker.to(args.device)

    arcface.to(args.device, dtype=weight_dtype)
    generator.to(args.device, dtype=weight_dtype)
    reconstructor.to(args.device, dtype=weight_dtype)
    diffusion.to(args.device, dtype=weight_dtype)
    heatmap.to(args.device)
    hopenet.to(args.device, dtype=weight_dtype)
    threedmm.to(args.device, dtype=weight_dtype)

    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                           T.Resize(256)])

    os.makedirs(args.reconstruction_path, exist_ok=True)
    os.makedirs(args.deid_path, exist_ok=True)
    os.makedirs(args.results_path, exist_ok=True)

    ds = GenerateEvalDataset(root_dir=args.data_path, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # Prepare evaluation objects
    gaze_l2 = MetricCollector(compute_fn=l2_error)
    pose_l2 = MetricCollector(compute_fn=l2_error)
    pose_hope_l2 = MetricCollector(compute_fn=l2_error)
    expression_l2 = MetricCollector(compute_fn=l2_error)

    for idx, data in tqdm(enumerate(dl), total=len(dl)):
        with torch.no_grad():
            with (torch.cuda.amp.autocast(dtype=weight_dtype)):
                current_target, file_names, current_ids = data
                current_target = current_target.to(args.device, weight_dtype)

                arc_id_target = arcface(id_transform(current_target))
                if args.itm:
                    arc_id_target = face_tracker(arc_id_target)

                deid, mask = generator(current_target, arc_id_target)  # TODO: DOUBLE CHECK THIS
                deid = deid * mask + current_target * (1 - mask)
                deid.clamp(-1, 1)

                if args.apply_noise or args.apply_diffusion:
                    for iterations in range(args.diffusion_iterations):
                        scheduler = SingleStepDPMSolverScheduler(num_timesteps=args.timestep_range,
                                                                 beta_start=0.00085,
                                                                 beta_end=0.012,
                                                                 solver_order=1,
                                                                 rescale_betas_zero_snr=False)

                        noise = torch.randn(deid.shape, device=deid.device)
                        noisy_deid = scheduler.add_noise(deid, noise, args.timestep_origin)
                        noisy_deid_prev = noisy_deid.clone()

                        if args.apply_diffusion:
                            t = torch.tensor(args.timestep_origin, device=noisy_deid.device).long()
                            timestep_batch = t.repeat(noisy_deid.shape[0])

                            noise_prediction = diffusion(noisy_deid, -1 * arc_id_target, timestep_batch)

                            noisy_deid = scheduler.step(noise_prediction, t, noisy_deid)

                        deid = noisy_deid * mask + current_target * (1 - mask)
                        deid = deid.clamp(-1, 1)

                reconstruction = reconstructor(deid)
                reconstruction = ((reconstruction.clamp(-1, 1).permute(0, 2, 3, 1) + 1) / 2) * 255.0
                reconstruction = np.clip(reconstruction.cpu().numpy(), 0, 255).astype('uint8')

                noise_prediction = ((noise_prediction.clamp(-1, 1).permute(0, 2, 3, 1) + 1) / 2) * 255.0
                noise_prediction = np.clip(noise_prediction.cpu().numpy(), 0, 255).astype('uint8')

                noisy_deid_prev = ((noisy_deid_prev.clamp(-1, 1).permute(0, 2, 3, 1) + 1) / 2) * 255.0
                noisy_deid_prev = np.clip(noisy_deid_prev.cpu().numpy(), 0, 255).astype('uint8')

                deid_eval = deid
                deid = ((deid.clamp(-1, 1).permute(0, 2, 3, 1) + 1) / 2) * 255.0
                deid = np.clip(deid.cpu().numpy(), 0, 255).astype('uint8')

                # pose evaluation
                yaw, pitch, roll = hopenet(hope_transform(deid_eval))

                yaw_predicted = softmax(yaw)
                pitch_predicted = softmax(pitch)
                roll_predicted = softmax(roll)

                deid_yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                deid_pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                deid_roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                deid_pose = torch.concat([deid_yaw_predicted[..., None],
                                          deid_pitch_predicted[..., None],
                                          deid_roll_predicted[..., None]], dim=1)

                yaw, pitch, roll = hopenet(hope_transform(current_target))

                yaw_predicted = softmax(yaw)
                pitch_predicted = softmax(pitch)
                roll_predicted = softmax(roll)

                target_yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                target_pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                target_roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                target_pose = torch.concat([target_yaw_predicted[..., None],
                                            target_pitch_predicted[..., None],
                                            target_roll_predicted[..., None]], dim=1)

                pose_hope_l2.update(target_pose, deid_pose)

                # Calculate gaze error
                hm_target = heatmap.get_preds_fromhm(gaze_transform(current_target))
                hm_deid = heatmap.get_preds_fromhm(gaze_transform(deid_eval))

                gaze_l2.update(hm_target, hm_deid)

                # Pose and expression
                deid_shape = threedmm(mm_transform(deid_eval))
                target_shape = threedmm(mm_transform(current_target))

                pose_l2.update(target_shape[3], deid_shape[3])
                expression_l2.update(target_shape[1], deid_shape[1])

                # Add batch to queue
                tensor_queue.put((reconstruction, deid, noisy_deid_prev, noise_prediction, current_ids, file_names))

    print('Data finished processing...')
    done_value.value = True
    for _ in range(args.num_processors):
        tensor_queue.put("terminate")

    print(args.reconstruction_path)
    print("Pose L2:", pose_l2.compute())
    print("Pose HopeNet L2:", pose_hope_l2.compute())
    print("Expression L2:", expression_l2.compute())
    print("Gaze L2:", gaze_l2.compute())

    result_dict = {}
    result_dict['pose_l2'] = pose_l2.compute().item()
    result_dict['pose_hopenet_l2'] = pose_hope_l2.compute().item()
    result_dict['expression_l2'] = expression_l2.compute().item()
    result_dict['gaze_l2'] = gaze_l2.compute().item()

    with open(f'{args.results_path}' + 'evaluation.json', 'w') as fp:
        json.dump(result_dict, fp)


def process_tensors(args, tensor_queue, done_value, pid=0):
    import os
    from PIL import Image

    print('Running process:', pid)

    while not done_value.value or not tensor_queue.empty():
        if not tensor_queue.empty():
            queue_item = tensor_queue.get()
            if queue_item == "terminate":
                break
            reconstruction, x_perturbed, noisy_deid_prev, noise_prediction, current_ids, file_names = queue_item
            for idx_im, (im, d_im, n_im, n) in enumerate(
                    zip(reconstruction, x_perturbed, noisy_deid_prev, noise_prediction)):

                current_id = current_ids[idx_im]
                file_name = file_names[idx_im]

                if not os.path.isdir(args.reconstruction_path + current_id):
                    os.mkdir(args.reconstruction_path + current_id)

                if not os.path.isdir(args.deid_path + current_id):
                    os.mkdir(args.deid_path + current_id)

                if args.noise_path:
                    os.makedirs(args.noise_path + current_id, exist_ok=True)
                    Image.fromarray(n).save(args.noise_path + current_id + '/' + file_name)

                if args.noisy_x_path:
                    os.makedirs(args.noisy_x_path + current_id, exist_ok=True)
                    Image.fromarray(n_im).save(args.noisy_x_path + current_id + '/' + file_name)

                Image.fromarray(im).save(args.reconstruction_path + current_id + '/' + file_name)

            del reconstruction
            del x_perturbed
            del noisy_deid_prev
            del noise_prediction
            del current_ids
            del file_names

    print('Batches finished processing...')


if __name__ == '__main__':
    import torch
    import os
    import argparse

    import torch.multiprocessing as mp

    '''
    Generates de-identified face pre and post reconstruction attack. Evaluates attributes pre reconstruction attack.
    Generated data can be used to evaluate FID and Identity Retrieval (pre) and Reconstruction Attack Identity
    Retrieval (post)
    '''

    parser = argparse.ArgumentParser()

    noise_strength = 0.1
    step_range = 1000

    noise = 'with_noise'
    diffusion = 'with_diffusion'
    suffix = f'target-blend-adjusted-SSDPM-{noise_strength}-noise-{step_range}-step_range-50k'
    data_folder = 'RA_v2_baseline_plus_addition_skips_attention_00-5_noise_diffusion_30_steps_v2'
    model_chkp = 50000
    dataset = 'faceforensic'
    parser.add_argument('--df', type=str, default=data_folder)
    parser.add_argument('--mc', type=int, default=model_chkp)

    parser.add_argument('--reconstructor_config', type=str, default="reconstructor_plus_addition_skips_attention")
    parser.add_argument('--generator_config', type=str, default="baseline_plus_addition_skips_attention")
    parser.add_argument('--diffusion_config', type=str, default="diffusion_plus_addition_skips_attention_mlp")

    parser.add_argument(
        "--generator_path",
        type=str,
        default="E:/Evaluation/ID_DIFF/evaluation/checkpoints/"
                "DeID_v2_baseline_plus_addition_skips_attention_gaze_loss_plus_eye_discrimination/"
                "50000_model_state.pth",
    )
    parser.add_argument(
        "--diffusion_path",
        type=str,
        default="E:/Evaluation/ID_DIFF/evaluation/checkpoints/"
                "DeID_diffusion_baseline_plus_addition_skips_attention_mlp_id_only_3/"
                "100000_model_state.pth",
    )
    parser.add_argument(
        "--reconstructor_path",
        type=str,
        default=f"E:/Evaluation/ID_DIFF/evaluation/checkpoints/"
                f"{data_folder}/"
                f"{model_chkp}_model_state.pth",
    )
    parser.add_argument(
        "--arcface_path",
        type=str,
        default="../../pretrained/arcface/backbone.pth",
    )
    parser.add_argument(
        "--threedmm_path",
        type=str,
        default="../../pretrained/model_3dmm/epoch_20.pth",
    )
    parser.add_argument(
        "--awing_path",
        type=str,
        default='../../pretrained/adaptive_wing/WFLW_4HG.pth',
    )
    parser.add_argument(
        "--hopenet_path",
        type=str,
        default='../../pretrained/hopenet/hopenet_robust_alpha1.pkl',
    )

    parser.add_argument('--apply_denoising', type=bool, default=False)
    parser.add_argument('--denoising_strength', type=float, default=1.5)

    parser.add_argument('--data_path', type=str,
                        default=f'F:/dataset/{dataset}/test_aligned/',
                        )
    parser.add_argument('--reconstruction_path', type=str,
                        default=f'E:/Evaluation/FIVA_GUARD/evaluation_{dataset}/'
                                f'DeIDv2_no_ITM_RA_ssdpm/post/{suffix}/',
                        )
    parser.add_argument('--deid_path', type=str,
                        default=f'E:/Evaluation/FIVA_GUARD/evaluation_{dataset}/'
                                f'DeIDv2_no_ITM_RA_ssdpm/pre/{suffix}/',
                        )
    parser.add_argument('--noise_path', type=str,
                        default=f'E:/Evaluation/FIVA_GUARD/evaluation_{dataset}/'
                                f'DeIDv2_no_ITM_RA_ssdpm/noise/{suffix}/',
                        )
    parser.add_argument('--noisy_x_path', type=str,
                        default=f'E:/Evaluation/FIVA_GUARD/evaluation_{dataset}/'
                                f'DeIDv2_no_ITM_RA_ssdpm/noisy_pre/{suffix}/',
                        )
    parser.add_argument('--results_path', type=str,
                        default=f'E:/Evaluation/FIVA_GUARD/evaluation_{dataset}/'
                                f'DeIDv2_no_ITM_RA_ssdpm/pre/results/{suffix}/',
                        )

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_type', type=str, default="float16")

    parser.add_argument('--itm', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--track', type=bool, default=False)

    # Data loader workers
    parser.add_argument('--num_workers', type=int, default=6)

    # Data writer workers
    parser.add_argument('--num_processors', type=int, default=6)

    # Diffusion / Noising parameters
    parser.add_argument('--apply_noise', type=bool, default=noise == 'with_noise')
    parser.add_argument('--apply_diffusion', type=bool, default=diffusion == 'with_diffusion')
    parser.add_argument('--timestep_origin', type=float, default=int(noise_strength * 1000))
    parser.add_argument('--timestep_range', type=float, default=1000)
    parser.add_argument('--diffusion_iterations', type=int, default=1)
    parser.add_argument('--shuffle_vector', type=bool, default=False)
    parser.add_argument('--random_margin', type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.reconstruction_path, exist_ok=True)
    os.makedirs(args.deid_path, exist_ok=True)

    tensor_queue = mp.Queue()
    done_value = mp.Value('i', False)

    # Spin up post processors
    tensor_processors = [mp.Process(target=process_tensors, args=(args, tensor_queue, done_value, p_idx))
                         for p_idx in range(args.num_processors)]

    for p in tensor_processors:
        p.start()

    process_data(args, tensor_queue, done_value)

    for p in tensor_processors:
        p.join()

    print('...')
