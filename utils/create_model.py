from models import ResNet, simplenet
import torch.nn as nn
import torch
import types
import timm
import os
from utils import utils

def model_sizes(args, layer):
    if args.dataset in ['imagenet', 'visdaC']:
        if layer == 0:
            channels, resolution = 64, 112
        if layer == 1:
            channels, resolution = 256, 56
        if layer == 2:
            channels, resolution = 512, 28
        if layer == 3:
            channels, resolution = 1024, 14
        if layer == 4:
            channels, resolution = 2048, 7

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if layer == 0:
            channels, resolution = 64, 32
        if layer == 1:
            channels, resolution = 256, 32
        if layer == 2:
            channels, resolution = 512, 16
        if layer == 3:
            channels, resolution = 1024, 8
        if layer == 4:
            channels, resolution = 2048, 4

    return channels, resolution


def get_part(model,layer):
    if layer == 1:
        extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1]
    elif layer == 2:
        extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2]
    elif layer == 3:
        extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2, model.net.layer3]
    elif layer == 4:
        extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2, model.net.layer3, model.net.layer4]
    return nn.Sequential(*extractor)


class ExtractorHead(nn.Module):
    def __init__(self, net, head):
        super(ExtractorHead, self).__init__()
        self.net = net
        self.head = head

    def forward(self, x, features=False, train=True, use_entropy=False, **kwargs):
        entropy = utils.Entropy()
        loss_ent = 0.0
        out, feature = self.net(x, feature=True)
        ssh_loss = self.head(feature, train=train)
        if use_entropy:
            loss_ent = entropy(out)
        if features:
            return out, feature, ssh_loss + loss_ent
        else:
            return out, ssh_loss + loss_ent

def visda_forward(self, x, feature=True):
    features = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    features.append(x)

    x = self.layer1(x)
    features.append(x)
    x = self.layer2(x)
    features.append(x)
    x = self.layer3(x)
    features.append(x)
    x = self.layer4(x)
    features.append(x)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    if feature:
        return x, features
    else:
        return x

# This is the modified forward_features method from the timm model (special case for CIFAR-10/100)
# Ignore the error, as the function checkpoint_seq is out of context, but is correct inside timm model
def create_model(args, device='cpu', weights=None):
    func_type = types.MethodType
    # Creating model based on dataset
    if args.dataset == 'visdaC':
        num_classes = 12
        net = timm.create_model('resnet50', features_only=True, pretrained=False)
        net.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        net.fc = nn.Linear(2048, num_classes)
        net.forward = func_type(visda_forward, net)
        weights_path = os.path.join(args.root,  'weights', 'resnet50.pth')
        pretraining = torch.load(weights_path)
        del pretraining['fc.weight']
        del pretraining['fc.bias']
        net.load_state_dict(pretraining, strict=False)
    elif args.dataset in ['cifar10','cifar100']:
        num_classes = 10 if args.dataset == 'cifar10' else 100
        net = timm.create_model('resnet50', features_only=True, pretrained=False)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        net.fc = nn.Linear(2048, num_classes)
        net.forward = func_type(visda_forward, net)
        #net = ResNet.resnet50(num_classes)
    ssh = simplenet.SimpleNet(args.layers, args.embed_size, std1=args.std, std2=args.std2, dataset=args.dataset, device=device).to(device)
    model = ExtractorHead(net, ssh)

    # Loading weights
    if weights is not None:
        model.load_state_dict(weights, strict=False)

    return model
