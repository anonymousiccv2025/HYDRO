import torch
import os
import tqdm
import argparse

from PIL import Image
from HYDRO.data.utils import norm_crop

import facer


def align_faces(args):

    images = args.data_path
    target = args.output_path

    face_detector = facer.face_detector('retinaface/resnet50', device=args.device)
    for identity in tqdm.tqdm(os.listdir(images)):
        print(identity)
        for face in os.listdir(images + identity):
            if face[-3:] == 'jpg':
                im_hwc = facer.read_hwc(images + identity + '/' + face)
                im_bchw = facer.hwc2bchw(im_hwc).to(device=args.device)

                try:
                    with torch.inference_mode():
                        faces = face_detector(im_bchw)

                    lm_align = faces['points'][0].cpu().numpy()

                    im_aligned = norm_crop(im_hwc.numpy(), lm_align,
                                           image_size=args.image_size,
                                           shrink_factor=args.shrink_factor)

                    os.makedirs(target + identity, exist_ok=True)

                    Image.fromarray(im_aligned).save(target + identity + '/' + face)
                except Exception as e:
                    print(e)


if __name__ == '__main__':

    '''
    Five-point landmark alignment of faces.
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default=f'F:/path/to/data/',
                        help='Expect folder structure: path/to/dataset/id_0, id_1, ..., id_n/im_0, im_1, ..., im_m'
                        )
    parser.add_argument('--output_path', type=str,
                        default=f'F:/path/to/aligned_data/',
                        help='Will match the folder structures of --data_path.'
                        )

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--shrink_factor', type=float, default=0.75,
                        help="The shrink_factor determines how much background to keep around the face. A shrink factor"
                             " of 0.75 means that if you central crop the image by 75% (keeping the central 75% "
                             "of the image) you will get a crop that corresponds to a shrink factor of 1.0. If "
                             "you use a shrink factor of 1.0, the resulting image will be aligned same as the "
                             "expected input for most face recognition models, ArcFace included.")

    args = parser.parse_args()


