from torchvision import models
import torch.nn as nn

def get_resnet18_finetune_net(out_num):
    model = models.resnet18(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    #model.fc = nn.Linear(model.fc.in_features, out_num)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )

    return model, in_num

def get_resnet34_finetune_net(out_num):
    model = models.resnet34(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    #model.fc = nn.Linear(model.fc.in_features, out_num)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )

    return model, in_num

def get_resnet50_finetune_net(out_num):
    model = models.resnet50(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    #model.fc = nn.Linear(model.fc.in_features, out_num)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )

    return model, in_num

def get_resnet101_finetune_net(out_num):
    model = models.resnet101(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    #model.fc = nn.Linear(model.fc.in_features, out_num)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )

    return model, in_num

def get_resnet152_finetune_net(out_num):
    model = models.resnet152(pretrained=True)
    in_num = 224

    for p in model.parameters():
        p.requires_grad = False

    #model.fc = nn.Linear(model.fc.in_features, out_num)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_num)
    )

    return model, in_num
