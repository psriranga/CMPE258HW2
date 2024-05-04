from __future__ import print_function, division
from PIL import Image  # can solve the error of Glibc
import configargparse  # pip install configargparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import warnings
import shutil

import PIL
import PIL.Image

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")

os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'  # setting the environment variable

CHECKPOINT_PATH = "./outputs"
CHECKPOINT_file = os.path.join(CHECKPOINT_PATH, 'checkpoint.pth.tar')

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)

from TorchClassifier.Datasetutil.Visutil import imshow, vistestresult, matplotlib_imshow
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel
from TorchClassifier.myTorchModels.TorchOptim import gettorchoptim
from TorchClassifier.myTorchModels.TorchLearningratescheduler import setupLearningratescheduler
from TorchClassifier.TrainValUtils import ProgressMeter, AverageMeter, accuracy

model = None
device = None

parser = configargparse.ArgParser(description='myTorchClassify')
parser.add_argument('--data_name', type=str, default='imagenet_blurred',
                    help='data name: imagenet_blurred, tiny-imagenet-200, hymenoptera_data, CIFAR10, MNIST, flower_photos')
parser.add_argument('--data_type', default='trainonly', choices=['trainonly', 'trainvalfolder', 'traintestfolder', 'torchvisiondataset'],
                    help='the type of data')
parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/ImageClassData",
                    help='path to get data')
parser.add_argument('--img_height', type=int, default=224,
                    help='resize to img height, 224')
parser.add_argument('--img_width', type=int, default=224,
                    help='resize to img width, 224')
parser.add_argument('--save_path', type=str, default='./outputs/',
                    help='path to save the model')

# network
parser.add_argument('--model_name', default='resnet50',
                    help='the network')
parser.add_argument('--model_type', default='ImageNet', choices=['ImageNet', 'custom'],
                    help='the network')
parser.add_argument('--torchhub', default='facebookresearch/deit:main',
                    help='the torch hub link')
parser.add_argument('--resume', default="outputs/imagenet_blurred_resnet50_0328/model_best.pth.tar", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='Pytorch', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Pytorch.')
parser.add_argument('--pretrained', default=True,
                    help='use pre-trained model')
parser.add_argument('--learningratename', default='StepLR',
                    choices=['StepLR', 'ConstantLR', 'ExponentialLR', 'MultiStepLR', 'OneCycleLR'],
                    help='learning rate name')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'adamresnetcustomrate'],
                    help='select the optimizer')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batchsize', type=int, default=128,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=40,
                    help='epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--classmap', default='TorchClassifier/Datasetutil/imagenet1000id2label.json', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--GPU', type=bool, default=False,
                    help='use GPU')
parser.add_argument('--gpuid', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--ddp', default=False, type=bool,
                    help='Use multi-processing distributed training.')
parser.add_argument('--TAG', default='0910',
                    help='setup the experimental TAG to differentiate different running results')
parser.add_argument('--reproducible', type=bool, default=False,
                    help='get reproducible results we can set the random seed for Python, Numpy and PyTorch')

args = parser.parse_args()

def save_checkpoint(state, is_best, path=CHECKPOINT_PATH):
    filename = os.path.join(path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, start_epoch=0, num_epochs=25,
                tensorboard_writer=None, profile=None, checkpoint_path='./outputs/'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss = [0.0 for i in range(0, num_epochs)]
    val_loss = [0.0 for i in range(0, num_epochs)]
    train_acc = [0.0 for i in range(0, num_epochs)]
    val_acc = [0.0 for i in range(0, num_epochs)]

    if profile is not None:
        profile.start()

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(dataloaders['train']),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss[epoch] = epoch_loss
                train_acc[epoch] = epoch_acc
            else:
                val_loss[epoch] = epoch_loss
                val_acc[epoch] = epoch_acc

            progress.display(epoch, phase, epoch_loss, epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model_name,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, path=checkpoint_path)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Loss/train', epoch_loss, epoch)
            tensorboard_writer.add_scalar('Loss/val', val_loss[epoch], epoch)
            tensorboard_writer.add_scalar('Acc/train', epoch_acc, epoch)
            tensorboard_writer.add_scalar('Acc/val', val_acc[epoch], epoch)

    if profile is not None:
        profile.stop()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss, train_acc, val_acc


def setup_model_optimizer_scheduler():
    global model, device

    # Detect if we have a GPU available
    if torch.cuda.is_available():
        if args.GPU:
            torch.cuda.set_device(args.gpuid)
        device = torch.device("cuda:{}".format(args.gpuid))
        print("Using GPU:", torch.cuda.get_device_name(args.gpuid))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model_ft = None
    input_size = 0

    if args.model_name == "resnet50":
        model_ft = models.resnet50(pretrained=args.pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.numclasses)
        input_size = 224
    elif args.model_name == "vgg16":
        model_ft = models.vgg16(pretrained=args.pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.numclasses)
        input_size = 224
    elif args.model_name == "wide_resnet":
        model_ft = models.wide_resnet50_2(pretrained=args.pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.numclasses)
        input_size = 224

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if args.optimizer == 'SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=args.weight_decay, amsgrad=False)
    elif args.optimizer == 'adamresnetcustomrate':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=args.weight_decay, amsgrad=False)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model_ft.load_state_dict(checkpoint['state_dict'])
            optimizer_ft.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler, input_size


def main():
    global args
    global CHECKPOINT_PATH

    # Setup TensorBoard writer
    tensorboard_writer = SummaryWriter(log_dir=CHECKPOINT_PATH)

    # Setup learning rate scheduler
    scheduler = setupLearningratescheduler(args.learningratename, args.lr)

    # Setup the model, optimizer, and scheduler
    model, criterion, optimizer, exp_lr_scheduler, input_size = setup_model_optimizer_scheduler()

    # Print the model architecture
    print(summary(model, input_size=(3, input_size, input_size), batch_size=args.batchsize))

    # Setup data loaders
    dataloaders, dataset_sizes, class_names = loadTorchdataset(args)

    # Train the model
    model, train_loss, val_loss, train_acc, val_acc = train_model(model, dataloaders, dataset_sizes,
                                                                  criterion, optimizer, exp_lr_scheduler,
                                                                  start_epoch=args.start_epoch,
                                                                  num_epochs=args.epochs,
                                                                  tensorboard_writer=tensorboard_writer,
                                                                  profile=None,
                                                                  checkpoint_path=args.save_path)

    # Visualize training and validation results
    matplotlib_imshow(train_loss, val_loss, train_acc, val_acc, args.epochs)

    # Visualize some predictions
    vistestresult(model, dataloaders['val'], class_names, device, num_images=6)

    # Close TensorBoard writer
    tensorboard_writer.close()


if __name__ == "__main__":
    main()
