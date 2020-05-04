# -*- coding: utf-8 -*-
# author: Dabin Cheng time: 2020/3/5

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchfusion_utils.metrics import Accuracy
from dateset import UCF101, ClipSubstractMean, RandomCrop, Rescale, ToTensor
from module.main_net import MainNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

batch_size = 4

root_list = '/Users/dabincheng/Downloads/data/UCF-24'
info_list = '/Users/dabincheng/Downloads/data/label/trianlist.txt'

myUCF101 = UCF101(info_list, root_list,
                  transform=transforms.Compose([ClipSubstractMean(), Rescale(), RandomCrop(), ToTensor()]))
dataloader = DataLoader(myUCF101, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

moudel = MainNet(input_dim=32, hidden_dim=32, lstm_kernel_size=(3,3),
                 num_layers=4, batch_size=batch_size, num_class=24)
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
        data, label = data.float(), label.long()-1
        data = data.permute(0, 2, 1, 3, 4)
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = moudel(data)
        loss = criteria(predictions, label)
        train_acc.update(predictions, label)
        loss.backward()
        expp = torch.softmax(predictions, dim=1)
        soft_out = expp
        confidence, clas = expp.topk(1, dim=1)
        optimizer.step()

        print(i_batch + 1, 'loss:{}\t'.format(loss.item()), 'train_acc:{}\n'.format(train_acc.getValue()),
              # 'pre:{}\t'.format(predictions.data), 'label:{}\n'.format(label),
              'confidence:{}\t'.format(soft_out),
              'clas:{}'.format(clas))

    return


if __name__ == '__main__':

    train()


