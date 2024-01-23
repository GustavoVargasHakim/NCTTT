import os
import timm
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
import numpy as np
from utils import utils, create_model, prepare_dataset
import copy


def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    '''-------------------Loading Dataset----------------------------'''
    if args.split == 'test' or args.dataset in ['cifar10','cifar100']:
        teloader, _ = prepare_dataset.prepare_test_data(args)
    else:
        teloader, _ = prepare_dataset.prepare_val_data(args)
    input_size, _ = next(enumerate(teloader))[1]
    args.input_size = input_size.size(-1)

    '''--------------------Loading Model-----------------------------'''
    print('Loading model')
    print('Dataset: ', args.dataset)
    print('Corruption: ', args.corruption if args.dataset in ['cifar10', 'cifar100'] else 'N/A')
    print('Training optimizer: ', args.optimizer)
    print('Adaptation optimizer: ', args.optim)
    print('Layers: ', args.layers)
    print('Std 1: ', args.std)
    print('Std 2: ', args.std2)

    if args.source:
        model = timm.create_model('resnet50', num_classes=12).cuda()
    else:
        model = create_model.create_model(args, device=device).to(device)
    path = utils.get_path(args, is_best=args.best)
    checkpoint = torch.load(os.path.join(args.root, args.dataset, 'weights', path))
    model.load_state_dict(checkpoint['state_dict'])

    state = copy.deepcopy(model.state_dict())
    print('Number of iterations:', args.niter)

    '''-------------------Optimizer----------------------------------'''
    if args.source:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
    else:
        extractor = create_model.get_part(model, args.layers[-1])
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(extractor.parameters(), lr=args.adapt_lr)
        else:
            optimizer = torch.optim.SGD(extractor.parameters(), lr=args.adapt_lr)

    '''--------------------Test-Time Adaptation----------------------'''
    print('Test-Time Adaptation')
    iteration = [1, 3, 5, 10, 15, 20, 50, 100]
    scores_before = []
    scores_after = []
    if args.niter in iteration and not args.use_mean:
        validation = args.val_times
        indice = iteration.index(args.niter)
        good_good_V = np.zeros([indice + 1, validation])
        good_bad_V = np.zeros([indice + 1, validation])
        bad_good_V = np.zeros([indice + 1, validation])
        bad_bad_V = np.zeros([indice + 1, validation])
        accuracy_V = np.zeros([indice + 1, validation])
        for val in range(validation):
            good_good = np.zeros([indice + 1, len(teloader.dataset)])
            good_bad = np.zeros([indice + 1, len(teloader.dataset)])
            bad_good = np.zeros([indice + 1, len(teloader.dataset)])
            bad_bad = np.zeros([indice + 1, len(teloader.dataset)])
            correct = np.zeros(indice + 1)
            for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                model.load_state_dict(state)
                if args.source:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
                else:
                    if args.optimizer == 'adam':
                        optimizer = torch.optim.Adam(extractor.parameters(), lr=args.adapt_lr)
                    else:
                        optimizer = torch.optim.SGD(extractor.parameters(), lr=args.adapt_lr)
                correctness, _ = utils.test_batch(model, inputs, labels, source=args.source)

                if args.adapt:
                    utils.adapt_batch(model, args.niter, inputs, optimizer, iteration, args.save_iter, train=False, q1=args.q, q2=1 - args.q, method=args.method, entropy=args.use_entropy)

                    for k in range(len(iteration[:indice + 1])):
                        ckpt = torch.load(args.save_iter + 'weights_iter_' + str(iteration[k]) + '.pkl')
                        model.load_state_dict(ckpt['weights'])
                        correctness_new, _ = utils.test_batch(model, inputs, labels, source=args.source, q1=args.q, q2=1 - args.q, method=args.method)
                        for i in range(len(correctness_new.tolist())):
                            if correctness[i] == True and correctness_new[i] == True:
                                good_good[k, i + batch_idx * args.batch_size] = 1
                            elif correctness[i] == True and correctness_new[i] == False:
                                good_bad[k, i + batch_idx * args.batch_size] = 1
                            elif correctness[i] == False and correctness_new[i] == True:
                                bad_good[k, i + batch_idx * args.batch_size] = 1
                            elif correctness[i] == False and correctness_new[i] == False:
                                bad_bad[k, i + batch_idx * args.batch_size] = 1
                else:
                    correct += correctness.sum().item()

            for k in range(len(iteration[:indice + 1])):
                correct[k] += np.sum(good_good[k,]) + np.sum(bad_good[k,])
                accuracy = correct[k] / len(teloader.dataset)
                good_good_V[k, val] = np.sum(good_good[k,])
                good_bad_V[k, val] = np.sum(good_bad[k,])
                bad_good_V[k, val] = np.sum(bad_good[k,])
                bad_bad_V[k, val] = np.sum(bad_bad[k,])
                accuracy_V[k, val] = accuracy

        for k in range(len(iteration[:indice + 1])):
            print('--------------------RESULTS----------------------')
            print('Perturbation: ', args.corruption)
            print('Number of iterations: ', iteration[k])
            print('Good first, good after: ', str(good_good_V[k,].mean()) + '+/-' + str(good_good_V[k,].std()))
            print('Good first, bad after: ', str(good_bad_V[k,].mean()) + '+/-' + str(good_bad_V[k,].std()))
            print('Bad first, good after: ', str(bad_good_V[k,].mean()) + '+/-' + str(bad_good_V[k,].std()))
            print('Bad first, bad after: ', str(bad_bad_V[k,].mean()) + '+/-' + str(bad_bad_V[k,].std()))
            print('Accuracy: ', str(np.round(accuracy_V[k,].mean()*100,2)) + '+/-' + str(np.round(accuracy_V[k,].std()*100,2)))

    else:
        validation = 1
        good_good_V = np.zeros([1, validation])
        good_bad_V = np.zeros([1, validation])
        bad_good_V = np.zeros([1, validation])
        bad_bad_V = np.zeros([1, validation])
        accuracy_V = np.zeros([1, validation])
        nb_iteration_V = np.zeros([1, validation])
        for val in range(validation):
            good_good = []
            good_bad = []
            bad_good = []
            bad_bad = []
            correct = 0
            nb_iteration = []
            for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                model.load_state_dict(state)
                correctness, _, _ = utils.test_batch(model, inputs, labels, source=args.source, q1=args.q, q2=1 - args.q, method=args.method)

                if args.adapt:
                    nb_iteration.append(utils.adapt_batch(model, args.niter, inputs, optimizer, iteration, args.save_iter, train=False, use_mean= args.use_mean, two_std=args.two_std, entropy=args.use_entropy))
                    correctness_new, _ = utils.test_batch(model, inputs, labels, source=args.source, q1=args.q, q2=1 - args.q, method=args.method)
                    for i in range(len(correctness_new.tolist())):
                        if correctness[i] == True and correctness_new[i] == True:
                            good_good.append(1)
                        elif correctness[i] == True and correctness_new[i] == False:
                            good_bad.append(1)
                        elif correctness[i] == False and correctness_new[i] == True:
                            bad_good.append(1)
                        elif correctness[i] == False and correctness_new[i] == False:
                            bad_bad.append(1)
                else:
                    correct += correctness.sum().item()

            correct += np.sum(good_good) + np.sum(bad_good)
            accuracy = correct / len(teloader.dataset)
            good_good_V[0, val] = np.sum(good_good)
            good_bad_V[0, val] = np.sum(good_bad)
            bad_good_V[0, val] = np.sum(bad_good)
            bad_bad_V[0, val] = np.sum(bad_bad)
            accuracy_V[0, val] = accuracy
            nb_iteration_V[0, val] = np.mean(nb_iteration)

        print('--------------------RESULTS----------------------')
        print('Perturbation: ', args.corruption)
        if args.adapt:
            print('Iteration: ', str(nb_iteration_V[0,].mean()) + '+/-' + str(nb_iteration_V[0,].std()))
            print('Good first, good after: ', str(good_good_V[0,].mean()) + '+/-' + str(good_good_V[0,].std()))
            print('Good first, bad after: ', str(good_bad_V[0,].mean()) + '+/-' + str(good_bad_V[0,].std()))
            print('Bad first, good after: ', str(bad_good_V[0,].mean()) + '+/-' + str(bad_good_V[0,].std()))
            print('Bad first, bad after: ', str(bad_bad_V[0,].mean()) + '+/-' + str(bad_bad_V[0,].std()))
            print('Accuracy: ', str(np.round(accuracy_V[0,].mean()*100,2)) + '+/-' + str(np.round(accuracy_V[0,].std()*100,2)))


if __name__ == '__main__':
    args = configuration.argparser()
    experiment(args)
