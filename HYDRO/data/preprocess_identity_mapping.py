
import torch.nn.functional as F
import argparse
import os
import tqdm
import json
import numpy as np


def id_transform(x, h=64, w=64, resize=112):
    return F.interpolate(x[:, :, h:-h, w:-w], [resize, resize])


def main(args):
    unique_id_dict = {}

    id_list_string = os.listdir(args.test_data_dir)

    for idx, identity in enumerate(id_list_string):
        faces = os.listdir(os.path.join(args.test_data_dir, identity))

        for _ in faces:
            unique_id_dict[identity] = identity

    with open(f"{args.result_dir}identity_mapping.json", 'w') as fp:
        json.dump(unique_id_dict, fp, indent=4)


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
