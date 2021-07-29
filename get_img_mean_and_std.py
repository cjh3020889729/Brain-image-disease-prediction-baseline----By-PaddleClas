import numpy as np
from PIL import Image
import os

# 获取测试图片的像素均值与方差
id_count = 0
means = []
stds = []
for _, _, files in os.walk('/home/aistudio/PaddleClas/dataset/val'):
    for f in files:
        img = Image.open('/home/aistudio/PaddleClas/dataset/val/' + f).convert('RGB')
        img = np.asarray(img)

        mean = img.mean() / 255.
        std = img.std() / 255.

        means.append(mean)
        stds.append(std)

        id_count += 1
print('Test Mean Pix Value: ', np.sum(means) / id_count)
print('Test Std Pix Value: ', np.sum(stds) / id_count)


# 获取训练MRI-AD图片的像素均值与方差
id_count = 0
means = []
stds = []
for _, _, files in os.walk('/home/aistudio/PaddleClas/dataset/train/MRI/AD'):
    for f in files:
        img = Image.open('/home/aistudio/PaddleClas/dataset/train/MRI/AD/' + f).convert('RGB')
        img = np.asarray(img)

        mean = img.mean() / 255.
        std = img.std() / 255.

        means.append(mean)
        stds.append(std)

        id_count += 1
print('Train MRI AD Mean Pix Value: ', np.sum(means) / id_count)
print('Train MRI AD Std Pix Value: ', np.sum(stds) / id_count)

# 获取训练MRI-NC图片的像素均值与方差
id_count = 0
means = []
stds = []
for _, _, files in os.walk('/home/aistudio/PaddleClas/dataset/train/MRI/NC'):
    for f in files:
        img = Image.open('/home/aistudio/PaddleClas/dataset/train/MRI/NC/' + f).convert('RGB')
        img = np.asarray(img)

        mean = img.mean() / 255.
        std = img.std() / 255.

        means.append(mean)
        stds.append(std)

        id_count += 1
print('Train MRI NC Mean Pix Value: ', np.sum(means) / id_count)
print('Train MRI NC Std Pix Value: ', np.sum(stds) / id_count)


# 获取训练PET-AD图片的像素均值与方差
id_count = 0
means = []
stds = []
for _, _, files in os.walk('/home/aistudio/PaddleClas/dataset/train/PET/AD'):
    for f in files:
        img = Image.open('/home/aistudio/PaddleClas/dataset/train/PET/AD/' + f).convert('RGB')
        img = np.asarray(img)

        mean = img.mean() / 255.
        std = img.std() / 255.

        means.append(mean)
        stds.append(std)

        id_count += 1
print('Train PET AD Mean Pix Value: ', np.sum(means) / id_count)
print('Train PET AD Std Pix Value: ', np.sum(stds) / id_count)
# 获取训练PET-NC图片的像素均值与方差
id_count = 0
means = []
stds = []
for _, _, files in os.walk('/home/aistudio/PaddleClas/dataset/train/PET/NC'):
    for f in files:
        img = Image.open('/home/aistudio/PaddleClas/dataset/train/PET/NC/' + f).convert('RGB')
        img = np.asarray(img)

        mean = img.mean() / 255.
        std = img.std() / 255.

        means.append(mean)
        stds.append(std)

        id_count += 1
print('Train PET NC Mean Pix Value: ', np.sum(means) / id_count)
print('Train PET NC Std Pix Value: ', np.sum(stds) / id_count)