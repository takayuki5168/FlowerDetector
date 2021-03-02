from torchvision import models
import torch.nn as nn

def get_vgg11_finetune_net(out_num):
    model = models.vgg11_bn(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifiers.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )
    return model, in_num

def get_vgg19_finetune_net(out_num):
    model = models.vgg19_bn(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifiers.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )
    return model, in_num
