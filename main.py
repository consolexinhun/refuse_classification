import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

import os, sys, time, csv, datetime, logging, argparse
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import split, label_list



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")

TrainLoader = split(dir="./data/train", batch_size=32, mode="train")
TestLoader =  split(dir="./data/test", batch_size=32, mode="test")
logging.info("数据集夹加载完成，训练集一共：{}条，测试集一共：{}条".format(len(TrainLoader.dataset), len(TestLoader.dataset)))

res_model = models.resnet101(pretrained=True)
model = nn.Sequential(
    *list(res_model.children())[:-1],
    Flatten(),
    nn.Linear(res_model.fc.in_features, 6)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
criteon = nn.CrossEntropyLoss().to(device)

min_loss = 100000

for epoch in range(30):
    for step, (x, y) in enumerate(TrainLoader):
        model.train()
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss = criteon(logits, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    logging.info("train>>>>>>epoch:{}, loss{}".format(epoch, loss.item()))

    if loss.item() < min_loss:
        min_loss = loss.item()
        state = {
            "net": model.state_dict(),
            "optimizer": optimizer,
            "epoch": epoch
        }
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        name = "save_model/{}_epoch_{}.check".format(current_time, epoch)
        torch.save(state, name)

        torch.save(state, "save_model/best_model.check")


keys, values = [], []
for i in range(len(TestLoader.dataset)):
    keys.append(i)

checkpoint = torch.load("save_model/best_model.check")
model.load_state_dict(checkpoint["net"])
model.eval()
for x in TestLoader:
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)

        predict = pred.cpu().numpy()
        for p in predict:
            label = label_list[int(p)]
            values.append(label)

with open("key.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(zip(keys, values))