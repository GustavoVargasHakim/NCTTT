import argparse

def argparser():
    parser = argparse.ArgumentParser()

    #Directories
    parser.add_argument('--root', type=str, default='/home/vhakim/scratch/Projects/NCTTT/', help='Base path')
    parser.add_argument('--dataroot', type=str, default='/home/davidoso/Documents/Data/')
    parser.add_argument('--save', type=str, default='work/', help='Path for base training weights')
    parser.add_argument('--save-iter', type=str, default='work/', help='Path for base training weights')

    #General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--print-freq', type=int, default=10, help='Number of epochs to print progress')

    #Model
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--eval', action='store_true', help='Using Eval at training')
    parser.add_argument('--load', type=str)

    #SimpleNet
    parser.add_argument('--patchsize', type=int, default=3, help='Patching size')
    parser.add_argument('--patchstride', type=int, default=1, help='Patching stride')
    parser.add_argument('--layers', type=int, nargs='+', default=[1], help='Layer blocks to put additional modules on (very common in TTT methods)')
    parser.add_argument('--embed-size', type=int, default=1536, help='Embedding size')
    parser.add_argument('--th-dsc', type=float, default=0.5, help='Threshold for discriminator')
    parser.add_argument('--std', type=float, default=0.1, help='Noise standard deviation')
    parser.add_argument('--std2', type=float, default=0.015, help='Noise standard deviation')
    parser.add_argument('--hidden', type=int, default=4, help='Noise standard deviation')

    #Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'imagenet', 'visdaC'))
    parser.add_argument('--target', type=str, default='cifar10')
    parser.add_argument('--workers', type=int, default=6, help='Number of workers for dataloader')

    #Source training
    parser.add_argument('--method', type=str, default='original', help='Type of task', choices=('original', 'margin', 'margin2', 'margin3', 'margin4','margin5','margin6','margin7'))
    parser.add_argument('--epochs', type=int, default=100, help='Number of base training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Manual epoch number for restarts')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for base training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight', type=float, default=1.0, help='Weight for loss function')
    parser.add_argument('--full', action='store_true', help='To use all the features')
    parser.add_argument('--separate', action='store_true', help='To use all the features')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer to use', choices=('sgd', 'adam'))
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay',  type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--evaluate', action='store_true', help='Evaluating on evaluation set')
    parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')
    parser.add_argument('--lbd', type=float, default=1.0, help='Lambda')

    #Test-Time Adaptation
    parser.add_argument('--adapt', action='store_true', help='To adapt or not')
    parser.add_argument('--source', action='store_true', help='To use source training')
    parser.add_argument('--use-entropy', action='store_true', help='To use entropy loss at test-time')
    parser.add_argument('--level', default=5, type=int)
    parser.add_argument('--corruption', default='gaussian_noise')
    parser.add_argument('--val-times', default=1, type=int)
    parser.add_argument('--split', default='eval', type=str, help='To use the evaluation set or the test set from VisdaC')
    parser.add_argument('--adapt-lr', default=0.00001, type=float)
    parser.add_argument('--optim', default='adam', type=str, help='Optimizer to use', choices=('sgd', 'adam'))
    parser.add_argument('--niter', default=50, type=int)
    parser.add_argument('--best', action='store_true', help='Using best pretraining weights or not')
    parser.add_argument('--K', type=int, default=10, help='Num of classes')
    parser.add_argument('--use-mean', action='store_true', help='Use mean to stop iterate')
    parser.add_argument('--two-std', action='store_true', help='Using two noises to create true and fake samples')

    #Distributed
    parser.add_argument('--distributed', action='store_true', help='Activate distributed training')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:3456', help='url for distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
    parser.add_argument('--world-size', type=int, default=1, help='Number of nodes for training')

    return parser.parse_args()
