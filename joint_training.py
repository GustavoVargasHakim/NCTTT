import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import math

import configuration
from utils import create_model, prepare_dataset, utils

best_uns = math.inf
lbd = 1

def main(args):
    global best_uns
    global lbd

    #-----------------DISTRIBUTED TRAINING------------------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    current_device = local_rank
    torch.cuda.set_device(current_device)
    if rank == 0:
        print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                            rank=rank)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    if rank == 0:
       print('From Rank: {}, ==> Making model..'.format(rank))
       print('Test on original data')
       print('Optimizer: ', args.optimizer)
       print('Layers: ', args.layers)
       print('Noise std1: ', args.std)
       print('Noise std2: ', args.std2)

    #------------------DATALOADERS--------------------------------------------------------------------------------------
    cudnn.benchmark = True
    trloader, trsampler, teloader, tesampler = prepare_dataset.prepare_train_data(args)
    if args.dataset in ['cifar10', 'cifar100']:
        args.corruption = 'original'
        teloader, tesampler = prepare_dataset.prepare_test_data(args)
    input_size, _ = next(enumerate(teloader))[1]
    args.input_size = input_size.size(-1)

    #------------------CREATE MODEL-------------------------------------------------------------------------------------
    model = create_model.create_model(args, device = 'cuda').cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.module.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.module.parameters(), args.lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if rank == 0:
        print('\t\tTrain Loss \t\t Sup Loss \t\t SSH Loss \t\t Train Accuracy \t\t Val Loss \t\t Val Accuracy')

    # ------------------JOINT TRAINING----------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        trsampler.set_epoch(epoch)
        tesampler.set_epoch(epoch)
        acc_train, loss_train, loss_sup, loss_ssh = train(model, optimizer, trloader, args)
        acc_val, loss_val = 0.0, 0.0
        if args.evaluate:
            acc_val, loss_val = validate(model, teloader, args)

        if rank == 0:
            print(('Epoch %d/%d:' % (epoch, args.epochs)).ljust(24) +
                      '%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (loss_train, loss_sup, loss_ssh, acc_train, loss_val, acc_val))

        if args.evaluate:
            is_best = loss_val < best_uns
        else:
            is_best = False
        best_uns = max(loss_val, best_uns)

        if rank == 0:
            dict = {'epoch': epoch + 1,
                            'arch': args.model,
                            'state_dict': model.module.state_dict(),
                            'best_uns': best_uns,
                            'optimizer': optimizer.state_dict(),
                        }
            utils.save_checkpoint(dict, is_best, args)

def train(model, optimizer, train_loader, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    sup_losses = utils.AverageMeter('Sup Loss', ':.4e')
    ssh_losses = utils.AverageMeter('SSH Loss', ':.4e')
    losses = utils.AverageMeter('SSH Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()
    cross_entropy = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        #Compute output and loss
        output, ssh_loss = model(images, train=True)
        sup_loss = cross_entropy(output, target)
        loss = sup_loss + args.weight*ssh_loss

        #Compute accuracy
        acc1 = utils.accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        sup_losses.update(sup_loss.item(), images.size(0))
        ssh_losses.update(ssh_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)

    return top1.avg, losses.avg, sup_losses.avg, ssh_losses.avg


def validate(model, val_loader, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()
    cross_entropy = nn.CrossEntropyLoss()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, ssh_loss = model(images)
            loss = cross_entropy(output, target) + ssh_loss

            # measure accuracy and record loss
            acc1 = utils.accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, losses.avg


if __name__=='__main__':
    args = configuration.argparser()
    main(args)
