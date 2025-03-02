import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from HYDRO.data.dataset import GenerateEvalDataset
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

    cosface_embeddings = np.load(opt.cosface_embeddings_path)
    arcface_embeddings = np.load(opt.arcface_embeddings_path)
    facenet_embeddings = np.load(opt.facenet_embeddings_path)
    adaface_embeddings = np.load(opt.adaface_embeddings_path)
    elasticface_embeddings = np.load(opt.elasticface_embeddings_path)
    identity_list = np.load(opt.identity_list_path)
    with open(opt.identity_mapping_path, "r") as f:
        identity_mapping = json.load(f)

    ds = GenerateEvalDataset(root_dir=opt.test_data_dir, transform=eval_transform,
                             cache_name=opt.data_cache, use_cache=False)
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

    num_matches_cosface_far_old = 0
    num_matches_arcface_far_old = 0
    num_matches_adaface_far_old = 0
    num_matches_elaface_far_old = 0

    total = 0

    data_info = opt.test_data_dir.split('/')[-2]

    results = {}

    for idx, data in tqdm(enumerate(di), total=ds.__len__() // opt.batch_size):
        with torch.no_grad():
            images, file_names, current_ids = data
            current_ids = list(current_ids)

            # We do this because while current ID may be 761, it can be the same identity as 012
            # This is adjusted in the identity list as well, so if argmin of the distance is 7610, which corresponds
            # to the first face for ID 761, the identity list will return 012. This ensures correct matching.
            mapped_ids = np.asarray([identity_mapping[cid] for cid in current_ids])

            images = images.to(opt.device)

            id_embeddings_cosface = CosFace(F.interpolate(images, size=112))
            id_embeddings_arcface = ArcFace(F.interpolate(images, size=112))
            id_embeddings_facenet = FaceNet(F.interpolate(images, size=160))
            id_embeddings_adaface = AdaFace(F.interpolate(images[:, [2, 1, 0], :, :], size=112))
            id_embeddings_elasticface = ElasticFace(F.interpolate(images, size=112))

            id_embeddings_cosface = id_embeddings_cosface.cpu().numpy()
            id_embeddings_arcface = id_embeddings_arcface.cpu().numpy()
            id_embeddings_facenet = id_embeddings_facenet.cpu().numpy()
            id_embeddings_adaface = id_embeddings_adaface.cpu().numpy()
            id_embeddings_elasticface = id_embeddings_elasticface.cpu().numpy()

            d_cosface, dis_cosface = nearest_cosine_distance(cosface_embeddings, id_embeddings_cosface)
            d_arcface, dis_arcface = nearest_cosine_distance(arcface_embeddings, id_embeddings_arcface)
            d_facenet, dis_facenet = nearest_l2_distance(facenet_embeddings, id_embeddings_facenet)
            d_adaface, dis_adaface = nearest_cosine_distance(adaface_embeddings, id_embeddings_adaface)
            d_elaface, dis_elaface = nearest_cosine_distance(elasticface_embeddings, id_embeddings_elasticface)

            # FAR 0.001
            matches_cosface = np.logical_and(identity_list[d_cosface] == mapped_ids, dis_cosface < 0.7812)
            matches_arcface = np.logical_and(identity_list[d_arcface] == mapped_ids, dis_arcface < 0.7629)
            matches_facenet = np.logical_and(identity_list[d_facenet] == mapped_ids, dis_facenet < 0.9368)
            matches_adaface = np.logical_and(identity_list[d_adaface] == mapped_ids, dis_adaface < 0.7999)
            matches_elaface = np.logical_and(identity_list[d_elaface] == mapped_ids, dis_elaface < 0.7999)

            num_matches_cosface_far_0_001 += np.sum(matches_cosface.astype('int32'))
            num_matches_arcface_far_0_001 += np.sum(matches_arcface.astype('int32'))
            num_matches_facenet_far_0_001 += np.sum(matches_facenet.astype('int32'))
            num_matches_adaface_far_0_001 += np.sum(matches_adaface.astype('int32'))
            num_matches_elaface_far_0_001 += np.sum(matches_elaface.astype('int32'))

            # FAR 0.0001
            matches_cosface = np.logical_and(identity_list[d_cosface] == mapped_ids, dis_cosface < 0.7184)
            matches_arcface = np.logical_and(identity_list[d_arcface] == mapped_ids, dis_arcface < 0.6931)
            matches_facenet = np.logical_and(identity_list[d_facenet] == mapped_ids, dis_facenet < 0.7051)
            matches_adaface = np.logical_and(identity_list[d_adaface] == mapped_ids, dis_adaface < 0.7487)
            matches_elaface = np.logical_and(identity_list[d_elaface] == mapped_ids, dis_elaface < 0.7602)

            num_matches_cosface_far_0_0001 += np.sum(matches_cosface.astype('int32'))
            num_matches_arcface_far_0_0001 += np.sum(matches_arcface.astype('int32'))
            num_matches_facenet_far_0_0001 += np.sum(matches_facenet.astype('int32'))
            num_matches_adaface_far_0_0001 += np.sum(matches_adaface.astype('int32'))
            num_matches_elaface_far_0_0001 += np.sum(matches_elaface.astype('int32'))

            # FAR 0.00001
            matches_cosface = np.logical_and(identity_list[d_cosface] == mapped_ids, dis_cosface < 0.6032)
            matches_arcface = np.logical_and(identity_list[d_arcface] == mapped_ids, dis_arcface < 0.6244)
            matches_facenet = np.logical_and(identity_list[d_facenet] == mapped_ids, dis_facenet < 0.5879)
            matches_adaface = np.logical_and(identity_list[d_adaface] == mapped_ids, dis_adaface < 0.6960)
            matches_elaface = np.logical_and(identity_list[d_elaface] == mapped_ids, dis_elaface < 0.6994)

            num_matches_cosface_far_0_00001 += np.sum(matches_cosface.astype('int32'))
            num_matches_arcface_far_0_00001 += np.sum(matches_arcface.astype('int32'))
            num_matches_facenet_far_0_00001 += np.sum(matches_facenet.astype('int32'))
            num_matches_adaface_far_0_00001 += np.sum(matches_adaface.astype('int32'))
            num_matches_elaface_far_0_00001 += np.sum(matches_elaface.astype('int32'))

            total += images.shape[0]

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

    print(f"ID retrieval 0.63 TH CosFace:"
          f" FAR 1e-5: {round(num_matches_cosface_far_old / total, 2)}")
    print(f"ID retrieval 0.63 TH ArcFace:"
          f" FAR 1e-5: {round(num_matches_arcface_far_old / total, 2)}")
    print(f"ID retrieval 0.63 TH AdaFace:"
          f" FAR 1e-5: {round(num_matches_adaface_far_old / total, 2)}")
    print(f"ID retrieval 0.63 TH ElasticFace:"
          f" FAR 1e-5: {round(num_matches_elaface_far_old / total, 2)}")

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

    with open(f'{opt.result_dir}{eval_dataset}_identity_retrieval_results.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Helper example of where precalculated embeddings are saved
    eval_dataset = 'faceforensic'   # or celeba
    data_set_path = f'F:/dataset/{eval_dataset}/test_meta/'

    # Paths
    parser.add_argument('--data_cache',
                        type=str,
                        default=None,)
    parser.add_argument('--test_data_dir',
                        type=str,
                        default=f'E:/path/to/data/to/evaluate/',
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
