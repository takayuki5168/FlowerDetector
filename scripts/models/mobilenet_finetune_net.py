from torchvision import models
import torch
import torch.nn as nn

def get_mobilenet_finetune_net(out_num):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    in_num = 256

    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(1280, 1000),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1000, 1000),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1000, out_num)
    )
    return model, in_num
