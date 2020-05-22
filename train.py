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
from module.clstm import ResCNNEncoder, DecoderRNN


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

num_class = 101
batch_size = 4
n_epochs = 20
milestones = [10, 15]
lr = 0.001

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512  # latent dim extracted by 2D CNN
res_size = 224  # ResNet image size
dropout_p = 0.0  # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

save_path = './save_module/'
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

# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
                            CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=num_class).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

# crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
#               list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
#               list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(crnn_params, lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)



def train_and_val(n_epochs, save_path):

    saving_criteria_of_module = 0
    train_loss_array = []
    val_loss_array =[]
    train_acc_array = []
    val_acc_array = []
    train_acc = Accuracy()
    val_acc =Accuracy(topK=1)

    cnn_encoder.train()
    rnn_decoder.train()

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
            optimizer.zero_grad()
            predictions = rnn_decoder(cnn_encoder(data))
            loss = criteria(predictions, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            total_train_data += label.size(0)
            train_acc.update(predictions, label)
            print('train', loss.item(), train_acc.getValue())
        scheduler.step()

        with torch.no_grad():

            cnn_encoder.eval()
            rnn_decoder.eval()

            val_acc.reset()
            for sample_batched in val_dataloader:
                data, label = sample_batched['video_x'], sample_batched['video_label']
                label = torch.squeeze(label)
                data, label = data.float(), label.long() - 1
                data, label = data.to(device), label.to(device)
                predictions = rnn_decoder(cnn_encoder(data))
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
            torch.save(cnn_encoder.state_dict(), os.path.join(save_path, 'cnn_encoder.pth'))
            torch.save(rnn_decoder.state_dict(), os.path.join(save_path, 'rnn_decoder.pth'))
            torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pth'))
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

