import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # learning-rate scheduler
import torchvision
import os
from exp import *
from dataset import *
from utils import * 

## parser
parser = argparse.ArgumentParser(description='Train model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-4, type=float, dest='lr')
parser.add_argument("--momentum", default=0.9, type=float, dest='momentum')
parser.add_argument("--batch_size", default=32, type=int, dest='batch_size')
parser.add_argument("--num_epoch", default=50, type=int, dest="num_epoch")
parser.add_argument("--base_dir", default="/content/drive/MyDrive/modeling", type=str, dest="base_dir")
parser.add_argument("--data_dir", default="/content/drive/MyDrive/DATA/OASIS/", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./drive/MyDrive/modeling/checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./drive/MyDrive/modeling/log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./drive/MyDrive/modeling/results", type=str, dest="result_dir")
parser.add_argument("--mode", default="train", type=str, dest="mode")
# parser.add_argument("--model_id", default="resnet", type=str, dest="model_id")
parser.add_argument("--network", default="resnet", choices=["vgg", "resnet", "densenet", "alexnet"], type=str, dest="network")
parser.add_argument("--learning_type", default=False, choices=[True, False], type=bool, dest="learning_type")
parser.add_argument("--layers", default=50, type=int, dest="layers")

args = parser.parse_args()
## hyperparameter

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
momentum = args.momentum

base_dir = args.base_dir
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

mode = args.mode

network = args.network
layers = args.layers
# model_id = args.model_id
learning_type = args.learning_type
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataloaders, dataset_sizes = dataloader(data_dir)

if 'resnet' in network: 
    if layers==50:
      model_together = torchvision.models.resnet50(pretrained=True)
    elif layers==101:
      model_together = torchvision.models.resnet101(pretrained=True)

    for param in model_together.parameters():
      param.requires_grad = learning_type # gradient 계산 추적을 하지 않도록 한다.
    
    num_ftrs = model_together.fc.in_features
    model_together.fc = nn.Linear(num_ftrs, 2) # 마지막 layer를 내 상황에 맞게 변경
    model_together = model_together.to(device)
    optimizer_together = optim.SGD(model_together.fc.parameters(), lr=lr, momentum=momentum)
elif 'vgg' in network:
    if layers==16:
      model_together = torchvision.models.vgg16_bn(pretrained=True)
    elif layers==19:
      model_together = torchvision.models.vgg19_bn(pretrained=True)
    for param in model_together.parameters():
      param.requires_grad = learning_type # gradient 계산 추적을 하지 않도록 한다.
    
    num_ftrs = model_together.classifier[6].in_features
    model_together.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 500),
                                             nn.Linear(500, 2)) 
    model_together = model_together.to(device)
    optimizer_together = optim.SGD(model_together.parameters(), lr=lr, momentum=momentum)
elif 'alexnet' in network:
    model_together = torchvision.models.alexnet(pretrained=True)
    for param in model_together.parameters():
      param.requires_grad = learning_type # gradient 계산 추적을 하지 않도록 한다.
    
    num_ftrs = model_together.classifier[6].in_features
    model_together.classifier[6] = nn.Linear(num_ftrs, 2)
    model_together = model_together.to(device)
    optimizer_together = optim.SGD(model_together.parameters(), lr=lr, momentum=momentum)
elif 'densenet' in network:
    if layers==161:
      model_together = torchvision.models.densenet161(pretrained=True)
    elif layers==201:
      model_together = torchvision.models.densenet201(pretrained=True)
    for param in model_together.parameters():
      param.requires_grad = learning_type # gradient 계산 추적을 하지 않도록 한다.
    
    num_ftrs = model_together.classifier.in_features
    model_together.classifier = nn.Linear(num_ftrs, 2)
    model_together = model_together.to(device)
    optimizer_together = optim.SGD(model_together.parameters(), lr=lr, momentum=momentum)
  
criterion = nn.MSELoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_together, step_size=7, gamma=0.1)
model_together = train_model_together(dataloaders, dataset_sizes, model_together, criterion, optimizer_together, exp_lr_scheduler, num_epochs=num_epoch)
net_name = "%s_%d_%d_%f" %(network, layers, num_epoch, lr)
save(result_dir, model_together, optimizer_together, net_name)
