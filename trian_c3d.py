# -*- coding: utf-8 -*-
# author: Dabin Cheng time: 2020/5/9


import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchfusion_utils.metrics import Accuracy
from data.ucf101 import UCF101, ClipSubstractMean, RandomCrop, Rescale, ToTensor, Normalize
from module.C3D import C3D


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

batch_size = 40
n_epochs = 20
milestones = [10, 15]

save_path = './save_module/0509_c3d_01.pt'
root_list = '/Users/dabincheng/downloads/UCF101_n_frames'
info_train_list = '/Users/dabincheng/downloads/ucfTrainTestlist/trainlist01.txt'
info_val_list = '/Users/dabincheng/downloads/ucfTrainTestlist/testlist001.txt'
train_data = UCF101(info_train_list, root_list,
                  transform=transforms.Compose([
                      Rescale(),
                      RandomCrop(),
                      ToTensor(),
                      Normalize()
                  ]))
val_data = UCF101(info_val_list, root_list,
                  transform=transforms.Compose([
                      Rescale(),
                      RandomCrop(),
                      ToTensor(),
                      Normalize()
                  ]))
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
module = C3D()

module = module.to(device)
lr = 0.001
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(module.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)


def train_and_val(n_epochs, save_path):

    saving_criteria_of_module = 0
    train_loss_array = []
    val_loss_array =[]
    train_acc_array = []
    val_acc_array = []
    train_acc = Accuracy()
    val_acc =Accuracy(topK=1)

    for i in range(n_epochs):
        train_loss = 0
        val_loss = 0
        total_train_data = 0
        total_val_data = 0
        train_acc.reset()

        for sample_batched in train_dataloader:
            data, label = sample_batched['video_x'], sample_batched['video_label']
            label = torch.squeeze(label)
            data, label = data.float(), label.long() - 1
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1, 3, 4)
            optimizer.zero_grad()
            predictions = module(data)
            loss = criteria(predictions, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            total_train_data += label.size(0)
            train_acc.update(predictions, label)
            print('train', loss.item(), train_acc.getValue())
        scheduler.step()

        with torch.no_grad():
            val_acc.reset()
            for sample_batched in val_dataloader:
                data, label = sample_batched['video_x'], sample_batched['video_label']
                label = torch.squeeze(label)
                data, label = data.float(), label.long() - 1
                data, label = data.to(device), label.to(device)
                data = data.permute(0, 2, 1, 3, 4)
                predictions = module(data)
                loss = criteria(predictions, label)
                val_loss += loss.item() * data.size(0)
                total_val_data += label.size(0)
                val_acc.update(predictions, label)
                print('val', loss.item(), val_acc.getValue())


        train_loss = train_loss / total_train_data
        val_loss = val_loss / total_val_data
        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)
        train_acc_array.append(train_acc.getValue())
        val_acc_array.append(val_acc.getValue())

        print(
            '{} / {} '.format(i + 1, n_epochs),
            'Train_loss: {}, '.format(train_loss),
            'val_loss: {}, '.format(val_loss),
            'Train_Accuracy: {}, '.format(train_acc.getValue()),
            'val_Accuracy: {}, '.format(val_acc.getValue())
        )

        if saving_criteria_of_module < val_acc.getValue():
            torch.save(module, save_path)
            saving_criteria_of_module = val_acc.getValue()
            print('--------------------------Saving Model---------------------------')

    plt.figure(figsize=(4, 4))
    x_axis = (range(n_epochs))
    plt.plot(x_axis, train_loss_array, 'r', val_loss_array, 'b')
    plt.title('A gragh of training loss vs validation loss')
    plt.legend(['train loss', 'val loss'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/0515_c3d_loss_01.png')
    plt.show()

    plt.figure(figsize=(4, 4))
    x_axis = (range(n_epochs))
    plt.plot(x_axis, train_acc_array, 'r', val_acc_array, 'b')
    plt.title('A gragh of training acc vs validation acc')
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('acc')
    plt.savefig('./result/0515_c3d_acc_01.png')
    plt.show()

    # x_axis = range(n_epochs)
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(x_axis, train_loss_array, 'b')
    # ax2.plot(x_axis, train_acc_array, 'r')
    #
    # ax1.set_xlabel('n_epochs')
    # ax1.set_ylabel('training_loss_array')
    # ax2.set_ylabel('training_acc_array')
    # plt.title('A gragh of training loss')
    # plt.savefig('/result/0509_01.png')
    # plt.show()

    return


if __name__ == '__main__':

    train_and_val(n_epochs, save_path)

