from __future__ import print_function
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os

from admm_core import Masking, CosineDecay, LinearDecay, add_sparse_args
from models import LeNet_5_Caffe, LeNet_300_100, VGG, VGG9
#Souvik: Following VGG16 model is imported separately from another file as the 
#model is for tiny-imagenet
from VGG_tiny_ImageNet import VGG_tiny
#souvik: Following import is for ResNet18 model import for CIFAR-10
from ResNet import ResNet18, ResNet50 
from utils import get_mnist_dataloaders, get_cifar10_dataloaders, \
                    get_cifar100_dataloaders, DatasetSplitter

from torchvision import datasets, transforms

#models = {}
#models['lenet5'] = (LeNet_5_Caffe)
def train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if lr_scheduler is not None:
            lr_scheduler.step()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.cross_entropy(output, target)
        #print(output.data)
        #'''
        if (epoch > args.start_admm_ep):
            mask.z_u_update(args, model, device, train_loader, optimizer, epoch, data, batch_idx)
            ce_loss, admm_loss, mixed_loss, sum_w, sum_u, sum_z = mask.append_admm_loss(args, model, ce_loss)
            print("Mixed loss value: ", mixed_loss)
            mixed_loss.backward()
        else:
            ce_loss.backward()
            admm_loss = 0.0
            sum_w = 0.0
            sum_u = 0.0
            sum_z = 0.0
            mixed_loss = ce_loss
        #'''
        #ce_loss.backward()
        with torch.no_grad():
            for name, W in model.named_parameters():
                if name=='features.20.weight' and epoch==10 and batch_idx==50:
                    torch.save(W.grad, '{}_{}_wBN_features.20.weight_grad.pt'.format(args.model_type,args.data))
                    print(W.grad.shape)
        if mask is not None:
            mask.step()
        else:
            optimizer.step()
    return ce_loss, admm_loss, mixed_loss, sum_w, sum_u, sum_z
            

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss=0
    total = 0
    correct = 0
    n = 0
    correct_5 = 0.0
    correct_1 = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
			#model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True) #index of max log-probability
			#correct += pred.eq(target.view_as(pred)).sum().item()
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            target = target.view(target.size(0),-1).expand_as(pred)
            correct = pred.eq(target).float()
            #compute top 5 correct
            correct_5 += correct[:, :5].sum()
            #compute top 1 correct
            correct_1 += correct[:, :1].sum()
            n += target.size(0)
    top1_acc = 100.*(correct_1/float(n))
    top5_acc = 100.*(correct_5/float(n))
    top1_err = 1 - (correct_1/float(n))
    top5_err = 1 - (correct_5/float(n))

    test_loss/=float(n)

    print('\n{}: Average loss: {:.4f}, Top1 Acc: {}/{} ({:.3f}%), Top5 Acc:{:.3f}%\n'.format('test evaluation' if is_test_set else 'Evaluation', \
				 test_loss, correct_1, n, 100.*correct_1/float(n), top5_acc))

    return top1_acc, top5_acc, top1_err, top5_err

def main():
	#Training settings
    parser =argparse.ArgumentParser(description='Pytorch MNIST example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
						help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_admm_ep', type=int, default=5, metavar='N',
                        help='when to start the admm based based dynamic L2 reg. loss on top of SGD loss(default: 5)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1000, metavar='S', help='random seed (default: 17)')
    #parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                    help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    #parser.add_argument('--save-model', type=str, default='./models/model.pt', help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='cifar10',
                        help='the dataset to be used for training [mnist, cifar10, tiny-imagenet]')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    #parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.0)
    parser.add_argument('--resume', type=str)
    #parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    #parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    #parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay_schedule', type=str, default='linear')
    #souvik: added the following argument for model selection
    parser.add_argument('--model_type', type=str, default='resnet18',
                         help = 'the models to be used for training [lenet5, lenet300, vgg16, resnet18, resnet50]')
    #core.add_sparse_args(parser)
    add_sparse_args(parser)

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #cuda_device = "cuda:"+args.gpu_idx
    #device = torch.device(cuda_device if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    if(args.data == 'mnist'):
    	train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)

    elif(args.data == 'cifar10'):
    	train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,\
            max_threads=args.max_threads)

    elif(args.data == 'cifar100'):
        train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split)

    elif(args.data == 'tiny_imagenet'):
        data_transforms = { 'train': transforms.Compose([transforms.RandomCrop(64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()]),
                            'val'  : transforms.Compose([transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),]) }

        data_dir = '/home/souvikku/tiny-imagenet-200/tiny-imagenet-200'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                            for x in ['train', 'val']}
        
        valid_loader = None
        if args.valid_split > 0.0:
            split = int(np.floor((1.0-args.valid_split) * len(image_datasets['train'])))
            train_dataset = DatasetSplitter(image_datasets['train'],split_end=split)
            val_dataset = DatasetSplitter(image_datasets['train'],split_start=split)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,\
                num_workers=4, pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,\
                num_workers=4, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, \
                shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=100, shuffle=True,\
                num_workers=4, pin_memory=True)
        

    #if args.model_type not in models:
    #	print('Please select a valid network model.\n')
    if args.model_type == 'lenet300':
        model = LeNet_300_100().to(device)
    if args.model_type == 'lenet5':
        model = LeNet_5_Caffe().to(device)
    if args.model_type == 'vgg16' and args.data == 'cifar10':
        model = VGG('VGG16', init_weights = True).to(device)
    if args.model_type == 'vgg16' and args.data == 'tiny_imagenet':
        model = VGG_tiny('VGG16', init_weights = True).to(device)
    #model = VGG9().to(device)
    if args.model_type == 'resnet18' and args.data == 'cifar10':
        model = ResNet18().to(device)
    if args.model_type == 'resnet50' and args.data == 'cifar100':
        model = ResNet50().to(device)
    if args.model_type == 'resnet18' and args.data == 'cifar100':
        model = ResNet18().to(device)
    print(model)
    
    torch.save(model.state_dict(),'init_model_{}_{}_dens{}.pt'.format(args.model_type, args.data, args.density))
    print('='*40)
    print('Prune mode: {}'.format(args.prune))
    print('Growth mode: {}'.format(args.growth))
    print('Redistribution mode: {}'.format(args.redistribution))
    print('='*40)
    # add custom prune/growth/redisribution here
    	#The following prune mode is not implemented yet
#    if args.prune == 'magnitude_variance':
#        print('Using magnitude-variance pruning. Switching to Adam optimizer...')
#        args.prune = magnitude_variance_pruning
#        args.optimizer = 'adam'
#    if args.redistribution == 'variance':
#        print('Using variance redistribution. Switching to Adam optimizer...')
#        args.redistribution = variance_redistribution
#        args.optimizer = 'adam'

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        print('using sgd as optim')
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    
    #Souvik: changed the gamma value from 0.1[deafult in the code] to 0.2
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_frequency, gamma=0.2)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[46875, 78125], gamma=0.2, last_epoch=-1)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[23437, 39062], gamma=0.2, last_epoch=-1)
    
    mask = None
    if not args.dense:
    	if args.decay_schedule == 'cosine':
    		decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
    	elif args.decay_schedule == 'linear':
    		decay = LinearDecay(args.prune_rate, len(train_loader)*(args.epochs))
    	print('using {} decay schedule'.format(args.decay_schedule))

    	mask = Masking(optimizer, decay, prune_rate=args.prune_rate, prune_mode=args.prune, \
    		growth_mode=args.growth, redistribution_mode=args.redistribution, verbose=args.verbose)
    	mask.add_module(model, density=args.density)

    best_testAcc = [0]
    traccr = []
    trloss = []
    valaccr_1 = [0]
    valaccr_5 = [0]
    valerr_1 = []
    valerr_5 = []
    
    ce_loss = []
    admm_loss = []
    mixed_loss = []
    
    sumW = []
    sumU = []
    sumZ = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        #print('start time', t0)
        ceLoss, admmLoss, mixedLoss, sum_w, sum_u, sum_z =  train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask)
        ce_loss.append(ceLoss)
        mixed_loss.append(mixedLoss)
        admm_loss.append(admmLoss)
        
        sumW.append(sum_w)
        sumU.append(sum_u)
        sumZ.append(sum_z)
        
        if args.valid_split > 0.0:
            val_acc = evaluate(args, model, device, valid_loader)
    
#        save_checkpoint({'epoch': epoch + 1,
#                         'state_dict': model.state_dict(),
#                         'optimizer' : optimizer.state_dict()},
#                        is_best=False, filename=args.save_model)
    	

        if not args.dense and epoch < args.epochs:
            #print('Inside at end of epoch: {} and {}'.format(args.dense, args.epochs))
            mask.at_end_of_epoch()
        #print('Validation acc:', val_acc)

        print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
        
        testAcc1, testAcc5, testErr1, testErr5 = evaluate(args, model, device, test_loader, is_test_set=True)
        #best_testAcc.append(evaluate(args, model, device, test_loader, is_test_set=True))
#        if (testAcc >= max(best_testAcc)):
#            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n". format(testAcc))
#            torch.save(model.state_dict(), "Best_model_{}_with_acc_{}.pt".format(args.data,testAcc))
#            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_testAcc)))
#            if (len(best_testAcc) > 1):
#                os.remove("Best_model_{}_with_acc_{}.pt".format(args.data, max(best_testAcc)))
#        print('Appending to best_testAcc {}'.format(testAcc))
#        best_testAcc.append(testAcc)
#        
        print("Finishing epoch: {}". format(epoch))
        if testAcc1 > max(best_testAcc):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n". format(testAcc1))
                torch.save(model.state_dict(), "{}_model_{}_dens{}_{}_epoch{}_testAcc_{}_seed{}.pt".format(args.model_type, args.data, args.density, args.prune, args.epochs, testAcc1, args.seed))
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_testAcc)))
                if len(best_testAcc) > 1:
                    os.remove("{}_model_{}_dens{}_{}_epoch{}_testAcc_{}_seed{}.pt".format(args.model_type, args.data, args.density, args.prune, args.epochs, max(best_testAcc), args.seed))
        best_testAcc.append(testAcc1)
        valaccr_5.append(testAcc5)
        valerr_1.append(testErr1)
        valerr_5.append(testErr5)
        
        

    print('End iteration.\n')
    with open('{}_density{}_{}_epochs{}_Top1testAcc_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in best_testAcc:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_TrainCELoss_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in ce_loss:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_TrainMixedLoss_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in mixed_loss:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_TrainADMMLoss_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in admm_loss:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_Top5testAcc_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in valaccr_5:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_Top1testErr_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in valerr_1:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_Top5testErr_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in valerr_5:
            f.write("%s\n" % item)


    with open('{}_density{}_{}_epochs{}_W_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in sumW:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_U_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in sumU:
            f.write("%s\n" % item)
    with open('{}_density{}_{}_epochs{}_Z_{}_seed{}.txt'.format(args.model_type, args.density, args.prune, args.epochs, args.data, args.seed), 'w') as f:
        for item in sumZ:
            f.write("%s\n" % item)
            
if __name__ == '__main__':
   main()


















