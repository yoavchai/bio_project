

#from torchsummary import summary
import torch
import torch.nn as nn
import unet
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import math
from data import MyDataset
import glob
import os
import tqdm
import argparse
from utils import save_images


checkpoint_path = os.path.join('checkpoint', 'best_checkpoint.pth.tar')

parser = argparse.ArgumentParser(description='TODO')
#parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=4, help='Number of epochs to train.')
#parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='The learning rate.')
parser.add_argument('--check_point_path', type=str, default=checkpoint_path, help='Path to checkpoint file')
parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=True, help='use pretrained')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Get a batch of training data
# inputs, masks = next(iter(dataloaders['train']))
#
# print(inputs.shape, masks.shape)
# for x in [inputs.numpy(), masks.numpy()]:
#     print(x.min(), x.max(), x.mean(), x.std())
#
# plt.imshow(reverse_transform(inputs[3]))
#
#
#
#
#
# model = pytorch_unet.UNet(6)
# model = model.to(device)
#
# summary(model, input_size=(3, 224, 224))


import numpy as np
def calc_loss(pred, target, metrics, bce_weight=0.1):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred) #TODO why it is not softmax??
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    #TODO check it is good
    # tp = np.logical_and(pred, target).sum()  # True positives
    # metrics['precision']  += tp / pred.sum()  # tp over all house predictions
    # metrics['recall'] += tp / target.sum()  # tp over all houses from ground truth

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


#need to input one image
def vis(pred,file_name):
    map = (pred[1] > pred[0]).to(dtype=torch.float32)
    save_images(map.unsqueeze(0),file_name)


def train_model(model, optimizer, scheduler, dataloaders,required_phanes, num_epochs=25,final_eval = False):
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in required_phanes:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for i, (inputs, labels) in enumerate( dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    if (final_eval):
                        vis(outputs[0],str(i) + "_ours_" + ".png")
                        vis(labels[0],str(i) + "_label" + ".png")

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()



                # statistics
                epoch_samples += inputs.size(0)

                if i % 20 == 0:
                    #print("i: ", i,"loss: ", metrics['loss'] / epoch_samples , "Dice: ",  metrics['dice'] / epoch_samples)
                    print('i: {} loss: {:.4f} dice: {:.4f}'.format(i,metrics['loss'] / epoch_samples,metrics['dice'] / epoch_samples))

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples



            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss and final_eval == False:
                print("saving best model")
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optim': optimizer}, checkpoint_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 2

model = unet.UNet(num_class).to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'],strict=True)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)


input_train_list = sorted( glob.glob(os.path.join('data','ct','train') + '/*.png') )
target_train_list = sorted( glob.glob(os.path.join('data','seg','train') + '/*.png') )

input_val_list = sorted( glob.glob(os.path.join('data','ct','val') + '/*.png') )
target_val_list = sorted( glob.glob(os.path.join('data','seg','val') + '/*.png') )


#datasets
trainset = MyDataset(input_train_list,target_train_list, test=False)
valset = MyDataset(input_val_list,target_val_list, test=True)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=4)

dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['val'] = val_loader

#train
required_phanes = ['train', 'val']
if (args.start_epoch< args.epochs):
    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders , required_phanes ,num_epochs=args.epochs - args.start_epoch)

required_phanes = ['val']
model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders ,required_phanes, num_epochs=1,final_eval = True)


