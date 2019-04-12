import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torchvision.utils import  save_image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet
from dataloader import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_net(net,
              epochs=100,
              iterations=100,
              data_dir='inpainting_set/',
              n_channels=3,
              lr=0.01,
              gpu=False):
    loader = DataLoader(data_dir)
    save_epochs = [0, 4, 9, 49, 99]
    optimizer = optim.SGD(net.parameters(),
                            lr=lr,
                            momentum=0.99,
                            weight_decay=0.0005)

    # optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        epochs_loss = 10
        loader.setMode('train')
        epoch_loss = 0
        for iteration in range(iterations):
            for i, (imgnp, labelnp, img, label) in enumerate(loader):
                shape = img.shape
                img_torch = torch.from_numpy(img.reshape(1,4,shape[1],shape[2])).float()
                label_torch = torch.from_numpy(label.reshape(1,3,shape[1],shape[2])).float()
                if gpu:
                    img_torch = img_torch.cuda()
                    label_torch = label_torch.cuda()
                net = net.to(device)
                pred = net(img_torch)
                loss = MSELoss(pred, label_torch)
                epoch_loss += loss.item()
                # optimize weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == 15 and iteration == 99 and epoch in save_epochs:
                    plt.imsave('samples/train_groundtruth%d.png' % (epoch + 1), labelnp)
                    plt.imsave('samples/train_input%d.png' % (epoch + 1), imgnp[:, :, :3])
                    save_image(pred, 'samples/train_output%d.png' % (epoch + 1))
        current_epochs_loss = epoch_loss/(16*iterations)
        if(current_epochs_loss < epochs_loss):
            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/BestModelCP%d.pth' % (current_epochs_loss))
            epochs_loss = current_epochs_loss
            print('Best model saved')
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, current_epochs_loss))
        if epoch in save_epochs:
            loader.setMode('test')
            net.eval()
            with torch.no_grad():
                for i,(imgnp, labelnp, img, label) in enumerate(loader):
                    shape = img.shape
                    img_torch = torch.from_numpy(img.reshape(1, 4, shape[1], shape[2])).float()
                    if gpu:
                        img_torch = img_torch.cuda()
                    pred = net(img_torch)
                    if i == 15:
                        plt.imsave('samples/test_groundtruth%d.png' % (epoch + 1), labelnp)
                        plt.imsave('samples/test_input%d.png' % (epoch + 1), imgnp[:, :, :3])
                        save_image(pred, 'samples/test_output%d.png' % (epoch+1))
    end = time.time()
    print('time cost is: ', end - start)

def MSELoss(pred_label, target_label):
    diff = pred_label - target_label
    diff = torch.pow(diff, 2)
    return torch.mean(diff)

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='inpainting_set/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    start = time.time()
    print('start time counting')
    args = get_args()

    net = UNet(n_channels=3)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        gpu=args.gpu,
        data_dir=args.data_dir)
