import os, sys
import shutil
import subprocess

import tqdm

import torchvision
from torchvision import models, transforms
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.vgg11_finetune_net import *
from models.vgg19_finetune_net import *

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
#dataset_dir = script_dir + '/../test_dataset/'
dataset_dir = script_dir + '/../dataset/'

subprocess.Popen(['tensorboard', '--logdir="logs/"'])


class Trainer(object):
    def __init__(self):
        self.cuda = True

        self.train_ratio = 0.8
        self.batch_size = 64
        self.max_epoch = 50
        self.out_num = len([f for f in os.listdir(dataset_dir) if os.path.isdir(dataset_dir + f)])

        self.load_model()
        self.load_dataset()

    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.Resize((self.in_num, self.in_num)),
             transforms.RandomHorizontalFlip(),
             #transforms.RandomResizedCrop(size=self.in_num, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=2),
             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)

        train_size = int(self.train_ratio * len(data))
        val_size = len(data) - train_size
        data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        self.validate_loader = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=False)

    def load_model(self):
        #self.model, self.in_num = get_vgg11_finetune_net(self.out_num)
        self.model, self.in_num = get_vgg19_finetune_net(self.out_num)

        if self.cuda:
            self.model = self.model.cuda()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        for batch_idx, (data, label) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train epoch=%d' % epoch):
            if self.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            self.optimizer.zero_grad()
            output = self.model(data)

            # loss
            if self.cuda:
                criterion = nn.CrossEntropyLoss().cuda()
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            loss_data = loss.data.item()
            train_loss += loss_data / len(data)

            loss.backward()
            self.optimizer.step()

            # accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum()
            train_accuracy += 100 * correct / len(label)

        self.train_writer.add_scalar('loss', train_loss / len(self.train_loader), epoch)
        self.train_writer.add_scalar('accuracy', train_accuracy / len(self.train_loader), epoch)

        self.model.eval()
        validate_loss = 0
        validate_accuracy = 0
        for batch_idx, (data, label) in tqdm.tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='Validate epoch=%d' % epoch):
            if self.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            output = self.model(data)

            # loss
            if self.cuda:
                criterion = nn.CrossEntropyLoss().cuda()
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            loss_data = loss.data.item()
            validate_loss += loss_data / len(data)

            # accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum()
            validate_accuracy += 100 * correct / len(label)

        self.validate_writer.add_scalar('loss', validate_loss / len(self.validate_loader), epoch)
        self.validate_writer.add_scalar('accuracy', validate_accuracy / len(self.validate_loader), epoch)

    def train(self):
        shutil.rmtree(script_dir + '/../logs/')
        self.train_writer = SummaryWriter(log_dir=script_dir + '/../logs/train')
        self.validate_writer = SummaryWriter(log_dir=script_dir + '/../logs/validate')

        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in tqdm.trange(0, self.max_epoch, desc='Train'):
            self.train_epoch(epoch)
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), script_dir + '/models/weights/model{}.pth'.format(epoch))

        torch.save(self.model.state_dict(), script_dir + '/models/weights/model.pth')
        self.train_writer.close()
        self.validate_writer.close()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
