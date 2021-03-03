from torchvision import models
import torch
import torch.nn as nn

def get_inceptionv3_finetune_net(out_num):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    in_num = 299

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, out_num)

    return model, in_num

