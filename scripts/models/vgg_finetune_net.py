from torchvision import models
import torch.nn as nn

def get_vgg11_finetune_net(out_num):
    model = models.vgg11_bn(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_num)

    return model, in_num

def get_vgg19_finetune_net(out_num):
    model = models.vgg19_bn(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_num)

    return model, in_num
