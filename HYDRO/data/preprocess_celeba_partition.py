import os
import tqdm

from PIL import Image

device = 'cuda:1'

images = "E:/Dataset/celeba/aligned/"
test_target = "E:/Dataset/celeba/test_aligned/"
val_target = "E:/Dataset/celeba/val_aligned/"
train_target = "E:/Dataset/celeba/train_aligned/"
identity_annotations = ''

with open('E:/Dataset/celeba/list_eval_partition.txt') as f:
    lines = f.readlines()

partition_dict = {line.split(" ")[0]: line.split(" ")[1][:-1] for line in lines}

for identity in tqdm.tqdm(os.listdir(images)):
    for face in os.listdir(images + identity):
        if face[-3:] == 'jpg':
            try:
                img = Image.open(images + identity + f"/{face}")
                partition = partition_dict[face]

                if partition == '0':
                    os.makedirs(train_target + identity, exist_ok=True)
                    img.save(train_target + identity + f"/{face}")
                elif partition == '1':
                    os.makedirs(val_target + identity, exist_ok=True)
                    img.save(val_target + identity + f"/{face}")
                elif partition == '2':
                    os.makedirs(test_target + identity, exist_ok=True)
                    img.save(test_target + identity + f"/{face}")
            except Exception as e:
                print(e, identity, face)
