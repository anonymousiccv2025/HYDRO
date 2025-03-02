
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import os

from HYDRO.data.dataset import LFWProtocolDatasetGenuine
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


def nearest_cosine_distance_lfw(u, v):
    u_n = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v_n = v / np.linalg.norm(v, axis=-1, keepdims=True)

    d = 1 - np.sum(u_n * v_n, axis=-1)

    return d


def nearest_l2_distance(u, v):
    u_n = np.sum(u ** 2, axis=1, keepdims=True)
    v_n = np.sum(v ** 2, axis=1, keepdims=True)

    d = np.sqrt(-2 * u.dot(v.T) + u_n + v_n.T)

    return np.argmin(d, axis=0), np.min(d, axis=0)


def nearest_l2_distance_lfw(u, v):
    u_n = np.sum(u ** 2, axis=1, keepdims=True)
    v_n = np.sum(v ** 2, axis=1, keepdims=True)

    d = np.sqrt(np.sum(np.square((u - v)), axis=-1))

    return d


def main(opt, eval_dataset):

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    eval_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(256, antialias=True),
                                         transforms.CenterCrop(192),
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

    ds = LFWProtocolDatasetGenuine(root_dir_genuine=opt.test_data_dir,
                                   root_dir_target=opt.gt_data_dir, transform=eval_transform,
                                   protocol_file="misc/pairsDevTest_genuine.txt")
    dl = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=opt.num_workers == 0)
    di = iter(dl)

    num_matches_cosface_far_0_001 = 0
    num_matches_arcface_far_0_001 = 0
    num_matches_facenet_far_0_001 = 0
    num_matches_adaface_far_0_001 = 0
    num_matches_elaface_far_0_001 = 0

    num_matches_cosface_far_0_0001 = 0
    num_matches_arcface_far_0_0001 = 0
    num_matches_facenet_far_0_0001 = 0
    num_matches_adaface_far_0_0001 = 0
    num_matches_elaface_far_0_0001 = 0

    num_matches_cosface_far_0_00001 = 0
    num_matches_arcface_far_0_00001 = 0
    num_matches_facenet_far_0_00001 = 0
    num_matches_adaface_far_0_00001 = 0
    num_matches_elaface_far_0_00001 = 0

    total = 0

    data_info = opt.test_data_dir.split('/')[-2]

    results = {}

    for idx, data in tqdm(enumerate(di), total=ds.__len__() // opt.batch_size):
        with torch.no_grad():
            genuine, target = data

            genuine = genuine.to(opt.device)
            target = target.to(opt.device)

            id_embeddings_cosface_genuine = CosFace(F.interpolate(genuine, size=112))
            id_embeddings_arcface_genuine = ArcFace(F.interpolate(genuine, size=112))
            id_embeddings_facenet_genuine = FaceNet(F.interpolate(genuine, size=160))
            id_embeddings_adaface_genuine = AdaFace(F.interpolate(genuine[:, [2, 1, 0], :, :], size=112))
            id_embeddings_elasticface_genuine = ElasticFace(F.interpolate(genuine, size=112))

            id_embeddings_cosface_genuine = id_embeddings_cosface_genuine.cpu().numpy()
            id_embeddings_arcface_genuine = id_embeddings_arcface_genuine.cpu().numpy()
            id_embeddings_facenet_genuine = id_embeddings_facenet_genuine.cpu().numpy()
            id_embeddings_adaface_genuine = id_embeddings_adaface_genuine.cpu().numpy()
            id_embeddings_elasticface_genuine = id_embeddings_elasticface_genuine.cpu().numpy()

            id_embeddings_cosface_target = CosFace(F.interpolate(target, size=112))
            id_embeddings_arcface_target = ArcFace(F.interpolate(target, size=112))
            id_embeddings_facenet_target = FaceNet(F.interpolate(target, size=160))
            id_embeddings_adaface_target = AdaFace(F.interpolate(target[:, [2, 1, 0], :, :], size=112))
            id_embeddings_elasticface_target = ElasticFace(F.interpolate(target, size=112))

            id_embeddings_cosface_target = id_embeddings_cosface_target.cpu().numpy()
            id_embeddings_arcface_target = id_embeddings_arcface_target.cpu().numpy()
            id_embeddings_facenet_target = id_embeddings_facenet_target.cpu().numpy()
            id_embeddings_adaface_target = id_embeddings_adaface_target.cpu().numpy()
            id_embeddings_elasticface_target = id_embeddings_elasticface_target.cpu().numpy()

            dis_cosface = nearest_cosine_distance_lfw(id_embeddings_cosface_genuine, id_embeddings_cosface_target)
            dis_arcface = nearest_cosine_distance_lfw(id_embeddings_arcface_genuine, id_embeddings_arcface_target)
            dis_facenet = nearest_l2_distance_lfw(id_embeddings_facenet_genuine, id_embeddings_facenet_target)
            dis_adaface = nearest_cosine_distance_lfw(id_embeddings_adaface_genuine, id_embeddings_adaface_target)
            dis_elaface = nearest_cosine_distance_lfw(id_embeddings_elasticface_genuine, id_embeddings_elasticface_target)

            # FAR 0.001 - threshold calculated from 'calculate_false_acceptance_rate_threshold_v2.py' for LFW dataset
            # with random pairs (no protocol)
            matches_cosface = np.where(dis_cosface < 0.7812, 1, 0)
            matches_arcface = np.where(dis_arcface < 0.7629, 1, 0)
            matches_facenet = np.where(dis_facenet < 0.9368, 1, 0)
            matches_adaface = np.where(dis_adaface < 0.7999, 1, 0)
            matches_elaface = np.where(dis_elaface < 0.7999, 1, 0)

            num_matches_cosface_far_0_001 += np.sum(matches_cosface.astype('int32'))
            num_matches_arcface_far_0_001 += np.sum(matches_arcface.astype('int32'))
            num_matches_facenet_far_0_001 += np.sum(matches_facenet.astype('int32'))
            num_matches_adaface_far_0_001 += np.sum(matches_adaface.astype('int32'))
            num_matches_elaface_far_0_001 += np.sum(matches_elaface.astype('int32'))

            # FAR 0.0001
            matches_cosface = np.where(dis_cosface < 0.7184, 1, 0)
            matches_arcface = np.where(dis_arcface < 0.6931, 1, 0)
            matches_facenet = np.where(dis_facenet < 0.7051, 1, 0)
            matches_adaface = np.where(dis_adaface < 0.7487, 1, 0)
            matches_elaface = np.where(dis_elaface < 0.7602, 1, 0)

            num_matches_cosface_far_0_0001 += np.sum(matches_cosface.astype('int32'))
            num_matches_arcface_far_0_0001 += np.sum(matches_arcface.astype('int32'))
            num_matches_facenet_far_0_0001 += np.sum(matches_facenet.astype('int32'))
            num_matches_adaface_far_0_0001 += np.sum(matches_adaface.astype('int32'))
            num_matches_elaface_far_0_0001 += np.sum(matches_elaface.astype('int32'))

            # FAR 0.00001
            matches_cosface = np.where(dis_cosface < 0.6032, 1, 0)
            matches_arcface = np.where(dis_arcface < 0.6244, 1, 0)
            matches_facenet = np.where(dis_facenet < 0.5879, 1, 0)
            matches_adaface = np.where(dis_adaface < 0.6960, 1, 0)
            matches_elaface = np.where(dis_elaface < 0.6994, 1, 0)

            num_matches_cosface_far_0_00001 += np.sum(matches_cosface.astype('int32'))
            num_matches_arcface_far_0_00001 += np.sum(matches_arcface.astype('int32'))
            num_matches_facenet_far_0_00001 += np.sum(matches_facenet.astype('int32'))
            num_matches_adaface_far_0_00001 += np.sum(matches_adaface.astype('int32'))
            num_matches_elaface_far_0_00001 += np.sum(matches_elaface.astype('int32'))

            total += genuine.shape[0]

    print(data_info)
    print(eval_dataset)
    print(f"ID retrieval CosFace:"
          f" FAR 1e-3: {round(num_matches_cosface_far_0_001 / total, 2)}"
          f" FAR 1e-4: {round(num_matches_cosface_far_0_0001 / total, 2)}"
          f" FAR 1e-5: {round(num_matches_cosface_far_0_00001 / total, 2)}")
    print(f"ID retrieval ArcFace:"
          f" FAR 1e-3: {round(num_matches_arcface_far_0_001 / total, 2)}"
          f" FAR 1e-4: {round(num_matches_arcface_far_0_0001 / total, 2)}"
          f" FAR 1e-5: {round(num_matches_arcface_far_0_00001 / total, 2)}")
    print(f"ID retrieval FaceNet:"
          f" FAR 1e-3: {round(num_matches_facenet_far_0_001 / total, 2)}"
          f" FAR 1e-4: {round(num_matches_facenet_far_0_0001 / total, 2)}"
          f" FAR 1e-5: {round(num_matches_facenet_far_0_00001 / total, 2)}")
    print(f"ID retrieval AdaFace:"
          f" FAR 1e-3: {round(num_matches_adaface_far_0_001 / total, 2)}"
          f" FAR 1e-4: {round(num_matches_adaface_far_0_0001 / total, 2)}"
          f" FAR 1e-5: {round(num_matches_adaface_far_0_00001 / total, 2)}")
    print(f"ID retrieval ElasticFace:"
          f" FAR 1e-3: {round(num_matches_elaface_far_0_001 / total, 2)}"
          f" FAR 1e-4: {round(num_matches_elaface_far_0_0001 / total, 2)}"
          f" FAR 1e-5: {round(num_matches_elaface_far_0_00001 / total, 2)}")

    results['CosFace_FAR_1e-3'] = round(num_matches_cosface_far_0_001 / total, 2)
    results['CosFace_FAR_1e-4'] = round(num_matches_cosface_far_0_0001 / total, 2)
    results['CosFace_FAR_1e-5'] = round(num_matches_cosface_far_0_00001 / total, 2)
    results['ArcFace_FAR_1e-3'] = round(num_matches_arcface_far_0_001 / total, 2)
    results['ArcFace_FAR_1e-4'] = round(num_matches_arcface_far_0_0001 / total, 2)
    results['ArcFace_FAR_1e-5'] = round(num_matches_arcface_far_0_00001 / total, 2)
    results['FaceNet_FAR_1e-3'] = round(num_matches_facenet_far_0_001 / total, 2)
    results['FaceNet_FAR_1e-4'] = round(num_matches_facenet_far_0_0001 / total, 2)
    results['FaceNet_FAR_1e-5'] = round(num_matches_facenet_far_0_00001 / total, 2)
    results['AdaFace_FAR_1e-3'] = round(num_matches_adaface_far_0_001 / total, 2)
    results['AdaFace_FAR_1e-4'] = round(num_matches_adaface_far_0_0001 / total, 2)
    results['AdaFace_FAR_1e-5'] = round(num_matches_adaface_far_0_00001 / total, 2)
    results['ElasticFace_FAR_1e-3'] = round(num_matches_elaface_far_0_001 / total, 2)
    results['ElasticFacet_FAR_1e-4'] = round(num_matches_elaface_far_0_0001 / total, 2)
    results['ElasticFace_FAR_1e-5'] = round(num_matches_elaface_far_0_00001 / total, 2)

    os.makedirs(opt.result_dir, exist_ok=True)

    with open(f'{opt.result_dir}{eval_dataset}_identity_retrieval_results_lfw_protocol.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Helper example of where precalculated embeddings are saved
    eval_dataset = 'lfw'
    data_set_path = f'F:/dataset/{eval_dataset}/test_meta/'

    # Paths
    parser.add_argument('--data_cache',
                        type=str,
                        default=None,)
    parser.add_argument('--test_data_dir',
                        type=str,
                        default=f'E:/path/to/data/to/evaluate/',
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')
    parser.add_argument('--gt_data_dir',
                        type=str,
                        default=f'E:/path/to/ground_truth/data/',
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')
    parser.add_argument('--result_dir',
                        type=str,
                        default=f'E:/path/where/we/save/results/',
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')

    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--cosface_path', type=str, default='../pretrained/cosface/'
                                                            'cosface_backbone.pth')
    parser.add_argument('--arcface_path', type=str, default='../pretrained/arcface/'
                                                            'arcface_backbone.pth')
    parser.add_argument('--adaface_path', type=str, default='../pretrained/adaface/'
                                                            'adaface_ir50_ms1mv2.ckpt')
    parser.add_argument('--elasticface_path', type=str, default='../pretrained/elasticface/'
                                                                '295672backbone.pth')

    parser.add_argument('--cosface_embeddings_path', type=str,
                        default=f'{data_set_path}cosface_embeddings.npy')
    parser.add_argument('--arcface_embeddings_path', type=str,
                        default=f'{data_set_path}arcface_embeddings.npy')
    parser.add_argument('--facenet_embeddings_path', type=str,
                        default=f'{data_set_path}facenet_embeddings.npy')
    parser.add_argument('--adaface_embeddings_path', type=str,
                        default=f'{data_set_path}adaface_embeddings.npy')
    parser.add_argument('--elasticface_embeddings_path', type=str,
                        default=f'{data_set_path}elasticface_embeddings.npy')

    parser.add_argument('--identity_list_path', type=str,
                        default=f'{data_set_path}identities.npy')
    parser.add_argument('--identity_mapping_path', type=str,
                        default=f'{data_set_path}identity_mapping.json')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    options = parser.parse_args()
    main(options, eval_dataset)
