import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torchvision

def save_checkpoint(state, is_best, args):
    string = '_'
    if args.separate:
        separate = '_separate'
    else:
        separate = ''
    for layer in args.layers:
        string += str(layer)
    if is_best:
        torch.save(state, args.save + args.dataset + '_' + args.model + '_' + args.optimizer + '_std1' + str(args.std) + '_std2' + str(args.std2) + '_lr' + str(args.lr) + string + separate + '_best.pth')
    else:
        torch.save(state, args.save + args.dataset + '_' + args.model + '_' + args.optimizer + '_std1' + str(args.std) + '_std2' + str(args.std2) + '_lr' + str(args.lr) + string + separate + '.pth')

def get_path(args, is_best=False):
    if args.separate:
        separate = '_separate'
    else:
        separate = ''
    if args.source:
        path = args.dataset + '_source.pth'
    else:
        string = '_'
        for layer in args.layers:
            string += str(layer)
        path = args.dataset + '_' + args.model + '_' + args.optimizer+ '_std1' + str(args.std) + '_std2' + str(args.std2) + '_lr' + str(args.lr) + string + separate
        if is_best:
            path += '_best.pth'
        else:
            path += '.pth'
    return path

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

'''--------------------Adaptation Function-----------------------------'''
def adapt_batch(net, niter, inputs, opt, iterations, save_iter, use_mean = False, train=False, **kwargs):
    entropy = kwargs['entropy']
    net.inference = False
    mean_global = 0.023
    net.train()
    for iteration in range(niter):
        _, loss = net(inputs, train=train, use_entropy=entropy)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if iteration + 1 in iterations and not use_mean:
            weights = {'weights': net.state_dict()}
            torch.save(weights, save_iter + 'weights_iter_' + str(iteration+1) +'.pkl')
        if use_mean:
            net.eval()
            _, scores = net(inputs, train=True)
            mean_scores = np.mean(scores)
            if mean_scores <= mean_global:
                return iteration
                break
            net.train()
    net.eval()

'''--------------------Testing Function-----------------------------'''
def test_batch(net, inputs, labels, source=False, full=False, **kwargs):
    net.eval()
    net.inference = False
    with torch.no_grad():
        if source:
            outputs = net(inputs)
        else:
            outputs, _ = net(inputs, train=False)
        acc = accuracy(outputs, labels)
        predicted = torch.argmax(outputs, dim=1)
        correctness = predicted.eq(labels).cpu()
    return correctness, acc

'''-------------------Loss Functions----------------------------------'''
class Entropy(torch.nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        return -(x.softmax(0)*x.log_softmax(0)).sum(0).mean()

'''-------------------Getting Adapters Parameters---------------------'''
def get_parameters(layers, model):
    parameters = []
    if layers[0] is not None:
        parameters += list(model.mask1.parameters())
    if layers[1] is not None:
        parameters += list(model.mask2.parameters())
    if layers[2] is not None:
        parameters += list(model.mask3.parameters())
    if layers[3] is not None:
        parameters += list(model.mask4.parameters())
    return parameters

def extractor_from_layer2(net):
    layers = [net.conv1,  net.bn1, nn.ReLU(inplace=True), net.layer1, net.layer2]
    return nn.Sequential(*layers)

def neg_log_likelihood_2d(target, z, log_det):
    log_likelihood_per_dim = target.log_prob(z) + log_det
    return -log_likelihood_per_dim.mean()

def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E
