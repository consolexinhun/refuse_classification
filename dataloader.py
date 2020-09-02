import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms, models
from PIL import Image

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visdom

trans = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],
    #                       [0.229,0.224,0.225])
])

label_list = ['plastic', 'trash', 'metal', 'glass', 'paper', 'cardboard']

class MyDataSet(Dataset):
    def __init__(self, mode, files, labels=None):
        """
        mode：模式train 或者test
        files:文件路径
        labels:文件的标签
        """
        super(MyDataSet, self).__init__()
        self.mode = mode
        self.files = files
        self.labels = labels
    def __getitem__(self, item):
        image_path = self.files[item]
        image = trans(image_path)
        if self.mode == "train":
            index = label_list.index(self.labels[item])
            label = torch.tensor(index)
            return image, label
        elif self.mode == "test":
            return image
    def __len__(self):
        return len(self.files)

def split(dir, batch_size, mode):
    files, labels = [], []
    if mode == "train":
        df = pd.read_csv("data/train.csv")
        if df.shape[0] != len(os.listdir(dir)):
            raise Exception("训练集的数量和train.csv中的数量不一致")

        train_files = os.listdir(dir)
        train_files.sort(key=lambda x: int(x[:-4]))
        for i, file_name in enumerate(train_files):
            file_path = os.path.join(dir, file_name)
            files.append(file_path)

            label = df.loc[df["filename"] == file_name].values[0, 1]
            labels.append(label)

        print(files[0])

        return DataLoader(MyDataSet("train", files, labels), batch_size=batch_size, shuffle=True)
            # print(df.loc[df["filename"] == file_name].values)  # 返回numpy数组，第一维是找到多少个（一般只有一个），第二维是header
            # if i == 3:
            #     break

    elif mode == "test":
        test_files = os.listdir(dir)
        test_files.sort(key=lambda x: int(x[:-4]))
        for i, file_name in enumerate(test_files):
            files.append(os.path.join(dir, file_name))
        return DataLoader(MyDataSet("test", files, labels), batch_size=batch_size)

if __name__ == '__main__':
    TrainLoader = split(dir="data/train", batch_size=32, mode="train")
    TestLoader  = split(dir="data/test", batch_size=32, mode="test")

    # print(len(TestLoader.dataset))
    # files = next(iter(TestLoader))
    #
    #
    # vis = visdom.Visdom() # 如果要查看图像，那么不能归一化
    # vis.images(files[0], win="x", opts=dict(title="x"))
    # vis.text(label_list[int(lables[0])], win='label', opts=dict(title="label"))





    # plt.figure()
    # sample = Image.open("data/train/0.jpg").convert('RGB')
    # tra = transforms.Compose([transforms.ToTensor()])
    # sample = tra(sample)
    # plt.imshow(sample)
    # plt.title(labels[0])
    # plt.show()



