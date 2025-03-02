def process_data(args):
    import torch
    import numpy as np
    import json

    from HYDRO.data.dataset import EvalDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchmetrics.image.fid import FrechetInceptionDistance

    import torch.nn.functional as F

    from tqdm import tqdm

    def fid_transform(x, h=32, w=32, resize=299):
        return F.interpolate(x[:, :, h:-h, w:-w], [resize, resize]) * 0.5 + 0.5

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

    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                           T.Resize(256)])

    os.makedirs(args.results_path, exist_ok=True)

    ds = EvalDataset(target_root_dir=args.source_data_path, source_root_dir=args.target_data_path, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # Prepare evaluation objects
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(args.device)

    for idx, data in tqdm(enumerate(dl), total=len(dl)):
        with torch.no_grad():
            with (torch.cuda.amp.autocast(dtype=weight_dtype)):
                current_target, deid_fid, file_names, current_ids = data
                current_target = current_target.to(args.device, weight_dtype)
                deid_fid = deid_fid.to(args.device, weight_dtype)

                # Add to FID and MFID calculators
                fid.update(fid_transform(current_target), real=True)
                fid.update(fid_transform(deid_fid), real=False)

    print(args.source_data_path)
    print("FID:", fid.compute())

    result_dict = {}
    result_dict['FID'] = fid.compute().item()

    with open(f'{args.results_path}' + 'fid_evaluation.json', 'w') as fp:
        json.dump(result_dict, fp)


if __name__ == '__main__':
    import torch
    import os
    import argparse

    parser = argparse.ArgumentParser()

    eval_dataset = 'celeba'
    source_data = "E:/Evaluation/FIVA_GUARD/evaluation_celeba/DeIDv2_no_ITM_RA_ssdpm/pre/target-blend-adjusted-SSDPM-0.05-noise-1000-step_range-50k/"

    parser.add_argument('--target_data_path', type=str,
                        default=f'F:/dataset/{eval_dataset}/test_aligned/',
                        )
    parser.add_argument('--source_data_path', type=str,
                        default=source_data
                        )

    parser.add_argument('--results_path', type=str,
                        default=f'E:/Evaluation/FIVA_GUARD/evaluation_{eval_dataset}/'
                                f'results/pre/DeIDv2_no_ITM_RA_ssdpm/pre/'
                                f'target-blend-adjusted-SSDPM-0.01-noise-1000-step_range-50k-lfw-v3/',
                        )

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_type', type=str, default="float16")

    # Data loader workers
    parser.add_argument('--num_workers', type=int, default=6)

    args = parser.parse_args()

    process_data(args)

