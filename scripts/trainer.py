import os, sys
import shutil

import tqdm

import torchvision
from torchvision import models, transforms

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from models.vgg_finetune_net import *
from models.mobilenet_finetune_net import *
from models.resnet_finetune_net import *

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
#dataset_dir = script_dir + '/../test_dataset/'
#dataset_dir = script_dir + '/../dataset_light/'
dataset_dir = script_dir + '/../dataset/'

class Trainer(object):
    def __init__(self, gpu, optimizer_type, lr, weight_decay):
        self.cuda = gpu if gpu >= 0 else -1

        self.model_architecture = 'resnet50'
        #self.model_architecture = 'vgg19'
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        print('[{} {}]'.format(self.model_architecture, self.optimizer_type))

        self.train_ratio = 0.8
        self.batch_size = 64
        self.max_epoch = 46
        self.out_num = len([f for f in os.listdir(dataset_dir) if os.path.isdir(dataset_dir + f)])

        self.load_model()
        self.load_dataset()

    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.Resize((self.in_num, self.in_num)),
             transforms.RandomHorizontalFlip(),
             #transforms.RandomResizedCrop(size=self.in_num, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=2),
             #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)

        train_size = int(self.train_ratio * len(data))
        val_size = len(data) - train_size
        data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        self.validate_loader = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=False)

    def load_model(self):
        if self.model_architecture == 'vgg11':
            self.model, self.in_num = get_vgg11_finetune_net(self.out_num)
        elif self.model_architecture == 'vgg19':
            self.model, self.in_num = get_vgg19_finetune_net(self.out_num)
        elif self.model_architecture == 'mobilenet':
            self.model, self.in_num = get_mobilenet_finetune_net(self.out_num)
        elif self.model_architecture == 'resnet18':
            self.model, self.in_num = get_resnet18_finetune_net(self.out_num)
        elif self.model_architecture == 'resnet50':
            self.model, self.in_num = get_resnet50_finetune_net(self.out_num)

        if self.cuda >= 0:
            self.model = self.model.to('cuda:{}'.format(self.cuda))

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        for batch_idx, (data, label) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train epoch={}'.format(epoch)):
            if self.cuda >= 0:
                data, label = data.to('cuda:{}'.format(self.cuda)), label.to('cuda:{}'.format(self.cuda))
            data, label = Variable(data), Variable(label)

            self.optimizer.zero_grad()
            output = self.model(data)

            # loss
            if self.cuda >= 0:
                criterion = nn.CrossEntropyLoss().to('cuda:{}'.format(self.cuda))
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

        self.train_writer.add_scalar('{}_loss/{}'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)), train_loss / len(self.train_loader), epoch)
        self.train_writer.add_scalar('{}_accuracy/{}'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)), train_accuracy / len(self.train_loader), epoch)

        self.model.eval()
        validate_loss = 0
        validate_accuracy = 0
        for batch_idx, (data, label) in tqdm.tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='Validate epoch={}'.format(epoch)):
            if self.cuda >= 0:
                data, label = data.to('cuda:{}'.format(self.cuda)), label.to('cuda:{}'.format(self.cuda))
            data, label = Variable(data), Variable(label)

            output = self.model(data)

            # loss
            if self.cuda >= 0:
                criterion = nn.CrossEntropyLoss().to('cuda:{}'.format(self.cuda))
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            loss_data = loss.data.item()
            validate_loss += loss_data / len(data)

            # accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum()
            validate_accuracy += 100 * correct / len(label)

        self.validate_writer.add_scalar('{}_loss/{}'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)), validate_loss / len(self.validate_loader), epoch)
        self.validate_writer.add_scalar('{}_accuracy/{}'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)), validate_accuracy / len(self.validate_loader), epoch)

    def train(self):
        # optimizer
        if self.optimizer_type == 'SGD':
            try:
                self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            except:
                self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'Adam':
            try:
                self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            except:
                self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # log name
        try:
            shutil.rmtree(script_dir + '/../logs/{}_{}/train'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)))
            shutil.rmtree(script_dir + '/../logs/{}_{}/validate'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)))
        except:
            pass
        self.train_writer = SummaryWriter(log_dir=script_dir + '/../logs/{}_{}/train'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)))
        self.validate_writer = SummaryWriter(log_dir=script_dir + '/../logs/{}_{}/validate'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay)))

        # model dir
        model_dir = script_dir + '/models/weights/{}_{}/'.format(self.model_architecture, '{}_lr{}_weightdecay{}'.format(self.optimizer.__class__.__name__, self.lr, self.weight_decay))
        try:
            shutil.rmtree(model_dir)
        except:
            pass
        os.mkdir(model_dir)

        for epoch in tqdm.trange(0, self.max_epoch, desc='Train'):
            self.train_epoch(epoch)
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), model_dir + '/model{}.pth'.format(epoch))

        torch.save(self.model.state_dict(), model_dir + '/model.pth')
        self.train_writer.close()
        self.validate_writer.close()

if __name__ == '__main__':
    gpu = int(sys.argv[sys.argv.index('-g') + 1]) if '-g' in sys.argv else -1

    optimizer_type = sys.argv[sys.argv.index('-o') + 1] if '-o' in sys.argv else 'Adam'
    lr = float(sys.argv[sys.argv.index('-l') + 1]) if '-l' in sys.argv else 1e-5
    weight_decay = float(sys.argv[sys.argv.index('-w') + 1]) if '-w' in sys.argv else 0

    trainer = Trainer(gpu, optimizer_type, lr, weight_decay)
    trainer.train()
