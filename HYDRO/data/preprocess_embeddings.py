
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import os

from HYDRO.data.dataset import GenerateEvalDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from HYDRO.model.adaface.load_model import load_pretrained_model
from HYDRO.model.arcface import iresnet50, iresnet100


def nearest_cosine_distance(u, v):
    u_n = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v_n = v / np.linalg.norm(v, axis=-1, keepdims=True)

    d = 1 - np.matmul(u_n, v_n.T)

    return np.argmin(d, axis=0), np.min(d, axis=0)


def nearest_l2_distance(u, v):
    u_n = np.sum(u ** 2, axis=1, keepdims=True)
    v_n = np.sum(v ** 2, axis=1, keepdims=True)

    d = np.sqrt(-2 * u.dot(v.T) + u_n + v_n.T)

    return np.argmin(d, axis=0), np.min(d, axis=0)


def main(opt):

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    eval_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop(288),
                                         transforms.Resize(160, antialias=True),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    CosFace = iresnet50()
    CosFace.load_state_dict(torch.load(opt.cosface_path))
    CosFace.eval()
    CosFace.to(device)

    ArcFace = iresnet100()
    ArcFace.load_state_dict(torch.load(opt.arcface_path))
    ArcFace.eval()
    ArcFace.to(device)

    FaceNet = InceptionResnetV1(pretrained='vggface2')
    FaceNet.eval()
    FaceNet.to(device)

    AdaFace = load_pretrained_model(opt.adaface_path)
    AdaFace.eval()
    AdaFace.to(device)

    ElasticFace = iresnet100(num_features=512)
    ElasticFace.load_state_dict(torch.load(opt.elasticface_path))
    ElasticFace.eval()
    ElasticFace.to(device)

    cosface_embeddings = None
    arcface_embeddings = None
    facenet_embeddings = None
    adaface_embeddings = None
    elasticface_embeddings = None

    ds = GenerateEvalDataset(root_dir=opt.test_data_dir, transform=eval_transform, use_cache=False,
                             cache_name=opt.data_cache)
    dl = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=opt.num_workers == 0)
    di = iter(dl)

    total = 0

    for idx, data in tqdm(enumerate(di), total=ds.__len__() // opt.batch_size):
        with torch.no_grad():
            images, _, _ = data

            images = images.to(opt.device)

            id_embeddings_cosface = CosFace(F.interpolate(images, size=112))
            id_embeddings_arcface = ArcFace(F.interpolate(images, size=112))
            id_embeddings_facenet = FaceNet(F.interpolate(images, size=160))
            id_embeddings_adaface = AdaFace(F.interpolate(images[:, [2, 1, 0], :, :], size=112))
            id_embeddings_elasticface = ElasticFace(F.interpolate(images, size=112))

            id_embeddings_cosface = id_embeddings_cosface
            id_embeddings_arcface = id_embeddings_arcface
            id_embeddings_facenet = id_embeddings_facenet
            id_embeddings_adaface = id_embeddings_adaface
            id_embeddings_elasticface = id_embeddings_elasticface

            if cosface_embeddings is None:
                cosface_embeddings = id_embeddings_cosface
                arcface_embeddings = id_embeddings_arcface
                facenet_embeddings = id_embeddings_facenet
                adaface_embeddings = id_embeddings_adaface
                elasticface_embeddings = id_embeddings_elasticface
            else:
                cosface_embeddings = torch.concat([cosface_embeddings, id_embeddings_cosface])
                arcface_embeddings = torch.concat([arcface_embeddings, id_embeddings_arcface])
                facenet_embeddings = torch.concat([facenet_embeddings, id_embeddings_facenet])
                adaface_embeddings = torch.concat([adaface_embeddings, id_embeddings_adaface])
                elasticface_embeddings = torch.concat([elasticface_embeddings, id_embeddings_elasticface])

            total += images.shape[0]

    np.save(opt.result_dir + 'cosface_embeddings.npy', cosface_embeddings.cpu().numpy())
    np.save(opt.result_dir + 'arcface_embeddings.npy', arcface_embeddings.cpu().numpy())
    np.save(opt.result_dir + 'facenet_embeddings.npy', facenet_embeddings.cpu().numpy())
    np.save(opt.result_dir + 'adaface_embeddings.npy', adaface_embeddings.cpu().numpy())
    np.save(opt.result_dir + 'elasticface_embeddings.npy', elasticface_embeddings.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Helper example of where precalculated embeddings are saved
    eval_dataset = 'faceforensic'
    data_set_path = f'F:/dataset/{eval_dataset}/test_meta/'

    # Paths
    parser.add_argument('--test_data_dir',
                        type=str,
                        default=f'E:/path/to/ground_truth/data/',
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')
    parser.add_argument('--result_dir',
                        type=str,
                        default=f'{data_set_path}',)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cosface_path', type=str, default='../pretrained/cosface/'
                                                            'cosface_backbone.pth')
    parser.add_argument('--arcface_path', type=str, default='../pretrained/arcface/'
                                                            'arcface_backbone.pth')
    parser.add_argument('--adaface_path', type=str, default='../pretrained/adaface/'
                                                            'adaface_ir50_ms1mv2.ckpt')
    parser.add_argument('--elasticface_path', type=str, default='../pretrained/elasticface/'
                                                                '295672backbone.pth')

    parser.add_argument('--data_cache', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    options = parser.parse_args()
    main(options)
