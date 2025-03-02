
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

import torch
import torch.nn.functional as F
import argparse
import os
import tqdm
import numpy as np
import json


def id_transform(x, h=64, w=64, resize=112):
    return F.interpolate(x[:, :, h:-h, w:-w], [resize, resize])


def main(args):

    identities = os.listdir(args.test_data_dir)

    with open(f"{data_set_path}identity_mapping.json", 'r') as fp:
        identity_mapping = json.load(fp)

    identity_array = []

    with torch.no_grad():
        for id in tqdm.tqdm(identities):
            aligned_frames = os.listdir(os.path.join(args.data_path, id))

            for _ in aligned_frames:
                identity_array.append(identity_mapping[id])

    identity_array = np.asarray(identity_array)

    np.save(f"{data_set_path}identities.npy", identity_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Helper example of where identity mapping are saved
    eval_dataset = 'faceforensic'
    data_set_path = f'F:/dataset/{eval_dataset}/test_meta/'

    # Paths
    parser.add_argument('--test_data_dir',
                        type=str,
                        default=f'E:/path/to/ground_truth/data/',
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')
    parser.add_argument('--result_dir',
                        type=str,
                        default=f'{data_set_path}', )

    parser.add_argument('--data_cache', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    main(args)
