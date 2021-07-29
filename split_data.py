import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 训练集与验证集的划分比例
split_rate = 0.8

train_root = 'PaddleClas/dataset/train'

img_kind = ['MRI', 'PET']
class2id = {'AD':0, 'NC':1}
class_kind = ['AD', 'NC']

train_data = []
for img_k in img_kind:
    for class_k in class_kind:
        img_root = os.path.join(train_root, img_k)
        img_dir_path = os.path.join(img_root, class_k)

        for _, _, files in os.walk(img_dir_path):
            for f in files:
                f_root = os.path.join(img_k, class_k)
                f = os.path.join(f_root, f)
                train_data.append([f, class2id[class_k]])

train_data = np.asarray(train_data)
np.random.shuffle(train_data)

train_d = train_data[:int(split_rate * len(train_data))]
eval_d  = train_data[int(split_rate * len(train_data)):]
all_d   = train_data

with open('PaddleClas/dataset/train/train_list.txt', 'w+') as f:
    train_tq = tqdm(enumerate(train_d))
    train_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(0, len(train_d)))
    for ids, i in train_tq:
        if ids + 1 < len(train_d):
            f.write('{0} {1}\n'.format(i[0], i[1]))
        else:
            f.write('{0} {1}'.format(i[0], i[1]))

        train_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(ids+1, len(train_d)))


with open('PaddleClas/dataset/train/val_list.txt', 'w+') as f:
    eval_tq = tqdm(enumerate(eval_d))
    eval_tq.set_description_str('-Eval Data Saved ({0}/{1})-'.format(0, len(eval_d)))
    for ids, i in eval_tq:
        if ids + 1 < len(eval_d):
            f.write('{0} {1}\n'.format(i[0], i[1]))
        else:
            f.write('{0} {1}'.format(i[0], i[1]))

        eval_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(ids+1, len(eval_d)))

with open('PaddleClas/dataset/train/all_list.txt', 'w+') as f:
    all_tq = tqdm(enumerate(all_d))
    all_tq.set_description_str('-All Data Saved ({0}/{1})-'.format(0, len(all_d)))
    for ids, i in all_tq:
        if ids + 1 < len(all_d):
            f.write('{0} {1}\n'.format(i[0], i[1]))
        else:
            f.write('{0} {1}'.format(i[0], i[1]))

        all_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(ids+1, len(all_d)))

print('Max Class Id: ', len(class_kind)-1)
print('Class Ids: ', [class2id[i] for i in class_kind])
print('Sample x data:\n', train_data[:3])