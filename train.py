# -*- coding: utf-8 -*-
# author: Dabin Cheng time: 2020/3/5

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchfusion_utils.metrics import Accuracy
from dateset import UCF101, ClipSubstractMean, RandomCrop, Rescale, ToTensor
from lstm import ConvLSTM
from res3D import Res3D
from moudel.C3D import C3D


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

batch_size = 4

root_list = '/Users/dabincheng/Downloads/data/UCF-24'
info_list = '/Users/dabincheng/Downloads/data/label/trianlist.txt'
myUCF101 = UCF101(info_list, root_list,
                  transform=transforms.Compose([ClipSubstractMean(), Rescale(), RandomCrop(), ToTensor()]))
dataloader = DataLoader(myUCF101, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# moudel = C3D(num_classes=101, pretrained=False)

moudel = ConvLSTM(input_dim=3, hidden_dim=3, kernel_size=(3,3), num_layers=4,
                 batch_size=4, seq_len=16, input_size=(112,112), num_class=24,
                 batch_first=False, bias=True, return_all_layers=False)

moudel = moudel.to(device)
lr = 0.001
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(moudel.parameters(), lr=lr)


def train():

    train_acc = Accuracy()
    train_acc.reset()

    for i_batch, sample_batched in enumerate(dataloader):
        data, label = sample_batched['video_x'], sample_batched['video_label']
        label = torch.squeeze(label)
        data, label = data.float(), label.long() - 1
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = moudel(data)
        loss = criteria(predictions, label)
        train_acc.update(predictions, label)
        loss.backward()
        # expp = torch.softmax(predictions, dim=1)
        # confidence, clas = expp.topk(1, dim=1)
        optimizer.step()

        print(i_batch + 1, 'loss:{}'.format(loss.item()), '\t', 'train_acc:{}'.format(train_acc.getValue()), '\n',
              # 'pre:{}'.format(predictions.data),
              # 'label:{}'.format(label),
              # 'confidence:{}'.format(confidence.data), '\t', 'clas:{}'.format(clas)
              )

    return


if __name__ == '__main__':

    train()

