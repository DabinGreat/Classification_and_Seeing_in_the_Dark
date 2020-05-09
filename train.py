# -*- coding: utf-8 -*-
# author: Dabin Cheng time: 2020/3/5

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchfusion_utils.metrics import Accuracy
# from dateset import UCF101, ClipSubstractMean, RandomCrop, Rescale, ToTensor
from dateset import UCF101, ClipSubstractMean, RandomCrop, Rescale, ToTensor
from module.C3D import C3D


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

batch_size = 4
n_epochs = 20
save_path = 'output_module.pt'

root_list = '/Users/dabincheng/Downloads/data/UCF-24'
info_list = '/Users/dabincheng/Downloads/data/label/trianlist.txt'
myUCF101 = UCF101(info_list, root_list,
                  transform=transforms.Compose([Rescale(), RandomCrop(), ToTensor()]))
dataloader = DataLoader(myUCF101, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
moudel = C3D()

# moudel = ConvLSTM(input_dim=3, hidden_dim=3, kernel_size=(3,3), num_layers=4,
#                  batch_size=4, seq_len=16, input_size=(112,112), num_class=24,
#                  batch_first=False, bias=True, return_all_layers=False)

moudel = moudel.to(device)
lr = 0.001
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(moudel.parameters(), lr=lr)
milestones = [10, 15]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

def train(n_epochs, save_path):

    saving_criteria_of_model = 0
    training_loss_array = []
    training_acc_array = []
    train_acc = Accuracy()

    for i in range(n_epochs):
        total_test_data = 0
        training_loss = 0
        train_acc.reset()

        for sample_batched in dataloader:
            data, label = sample_batched['video_x'], sample_batched['video_label']
            label = torch.squeeze(label)
            data, label = data.float(), label.long() - 1
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1, 3, 4)
            optimizer.zero_grad()
            predictions = moudel(data)
            loss = criteria(predictions, label)
            print(predictions.data, label) #
            loss.backward()
            # expp = torch.softmax(predictions, dim=1)
            # confidence, clas = expp.topk(1, dim=1)
            # print(expp.data, confidence, clas) #
            optimizer.step()
            training_loss += loss.item() * data.size(0)
            train_acc.update(predictions, label)
        scheduler.step()

        training_loss = training_loss / 2332
        training_loss_array.append(training_loss)
        training_acc_array.append(train_acc)

        print(
            '{} / {} '.format(i + 1, n_epochs),
            'Training loss: {}, '.format(training_loss),
            'Tran_Accuracy: {}, '.format(train_acc.getValue()))

        if saving_criteria_of_model < train_acc.getValue():
            torch.save(moudel, save_path)
            saving_criteria_of_model = train_acc.getValue()
            print('--------------------------Saving Model---------------------------')

    plt.figure(figsize=(4, 4))
    x_axis = range(n_epochs)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x_axis, training_loss_array, 'b')
    ax2.plot(x_axis, training_acc_array, 'r')

    ax1.set_xlabel('n_epochs')
    ax1.set_ylabel('training_loss_array')
    ax2.set_ylabel('training_acc_array')
    plt.title('A gragh of training loss')
    plt.savefig('a.png')
    plt.show()

    return


if __name__ == '__main__':

    train(n_epochs, save_path)

