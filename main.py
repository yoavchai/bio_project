

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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tqdm
import argparse
from utils import save_images, save_seg_images
import itertools
from networks import R2AttU_Net , AttU_Net
import os
import matplotlib.pyplot as plt


from torch.backends import cudnn
cudnn.benchmark = True

csv_path = os.path.join('checkpoint', 'training_stats.csv')

checkpoint_liver_path = os.path.join('checkpoint', 'liver_checkpoint.pth.tar')
checkpoint_lesion_path = os.path.join('checkpoint', 'lesion_checkpoint.pth.tar')

parser = argparse.ArgumentParser(description='TODO')
#parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=4, help='Number of epochs to train.')
#parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='The learning rate.')
#parser.add_argument('--check_point_path', type=str, default=checkpoint_path, help='Path to checkpoint file')
parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=True, help='use pretrained')
parser.add_argument('--train_mode', type=str, default='train_liver', help='options are train_liver/train_lesion/test')

args = parser.parse_args()


if args.train_mode not in ['train_liver','train_lesion','test']:
    AssertionError()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_loss(pred, target, metrics, bce_weight=0.5):
    #bce = 0
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred) #TODO why it is not softmax??
    #pred = torch.softmax(pred,dim=1)  # TODO why it is not softmax??
    dice = dice_loss(pred, target, metrics)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice_loss(pred, target,binary=True).data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    #TODO check it is good
    # tp = np.logical_and(pred, target).sum()  # True positives
    # metrics['precision']  += tp / pred.sum()  # tp over all house predictions
    # metrics['recall'] += tp / target.sum()  # tp over all houses from ground truth

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        if k == 'lesion_recall':
            outputs.append("{}: {:4f}".format(k, metrics[k] / metrics['lesion_recall_samples']))
        elif k =='lesion_precision':
            outputs.append("{}: {:4f}".format(k, metrics[k] / metrics['lesion_precision_samples']))
        else:
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def save_depth_maps(pred,file_name):
    threshold = -0.1
    # map = (pred[1] + threshold  > pred[0]).to(dtype=torch.float32) #TODO use global threshold
    # save_images(map.unsqueeze(0),file_name)
    pred_argmax = torch.argmax(pred, 1).float()
    depth_map = torch.floor(127.5*pred_argmax).long()
    image_dir = './result/depth_maps'
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    for i, img in enumerate(depth_map):
        plt.imsave(os.path.join(image_dir, file_name.replace('.png', '_'+str(i)+'.png')), img.cpu())

#need to input one image
def vis(pred,file_name):
    threshold = -0.1
    # map = (pred[1] + threshold  > pred[0]).to(dtype=torch.float32) #TODO use global threshold
    # save_images(map.unsqueeze(0),file_name)
    save_images(pred.unsqueeze(0),file_name)

def vis_seg(input, pred, pred_lesion,file_name):
    threshold = -0.1
    map = (pred[1] + threshold  > pred[0]).to(dtype=torch.float32) #TODO use global threshold
    lesion_map = (pred_lesion[1] + threshold  > pred_lesion[0]).to(dtype=torch.float32) #TODO use global threshold
    save_seg_images(input, map, lesion_map, file_name)


def train_model(model_livel,model_lesion, optimizer, scheduler, dataloaders,required_phanes, num_epochs=25,final_eval = False):
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

                model_livel.train()  # Set model to training mode
                model_lesion.train()
            else:
                model_livel.eval()  # Set model to evaluate mode
                model_lesion.eval()

            metrics_liver = defaultdict(float)
            metrics_lesion = defaultdict(float)
            epoch_samples = 0

            for i, (inputs, labels, image_before, image_after) in enumerate( dataloaders[phase]):
                inputs = inputs.to(device)
                image_before = image_before.to(device)
                image_after = image_after.to(device)
                labels = labels.to(device)
                #labels_lesion = labels_lesion.to(device)
                inputs = torch.cat([inputs, image_before, image_after],1)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if args.train_mode in ['train_lesion','test']:
                        with torch.no_grad():
                            model_livel.eval()
                            outputs = model_livel(inputs)
                            loss_liver = calc_loss(outputs, labels, metrics_liver) #to update metrics_liver
                            #loss_liver = 0
                    else:
                        outputs = model_livel(inputs)
                        loss_liver = calc_loss(outputs, labels, metrics_liver)

                    # if args.train_mode in ['train_lesion', 'test']:
                    #     #outpus_lesion = model_lesion(torch.cat((inputs,outputs),dim=1))
                    #     outpus_lesion = model_lesion(inputs)
                    #     loss_lesion = calc_loss(outpus_lesion, labels, metrics_lesion)
                    # else:
                    loss_lesion = 0
                    metrics_lesion['dice'] = 0
                    metrics_lesion['bce'] = 0
                    metrics_lesion['loss'] = 0

                    loss = loss_liver + loss_lesion

                    if (final_eval):
                        save_depth_maps(outputs, str(i) + "_depth_map.png")

                        vis(outputs[0,1,:,:],str(i) + "_liver_ours" + ".png")
                        vis(labels[0,1,:,:],str(i) + "_liver_label" + ".png")


                        vis(outputs[0,2,:,:],str(i) + "_lesion_ours" + ".png")
                        vis(labels[0,2,:,:],str(i) + "_lesion_label" + ".png")

                        # vis_seg(inputs[0, :3,:,:], labels[0,1,:,:], labels[0,2,:,:], str(i) + "_seg_label.png")
                        # vis_seg(inputs[0, :3,:,:], outputs[0,1,:,:], outputs[0,2,:,:], str(i) + "_seg_ours.png")

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()



                # statistics
                epoch_samples += inputs.size(0)

                if i % 20 == 0:
                    #print("i: ", i,"loss: ", metrics['loss'] / epoch_samples , "Dice: ",  metrics['dice'] / epoch_samples)
                    print('liver - i: {} loss: {:.4f} dice: {:.4f}'.format(i,metrics_liver['loss'] / epoch_samples,metrics_liver['dice'] / epoch_samples))
                    print('lesion - i: {} loss: {:.4f} dice: {:.4f}'.format(i, metrics_lesion['loss'] / epoch_samples,metrics_lesion['dice'] / epoch_samples))

            print_metrics(metrics_liver, epoch_samples, phase)
            print_metrics(metrics_lesion, epoch_samples, phase)

            if args.train_mode in ['train_liver']:
                epoch_loss = metrics_liver['dice'] / epoch_samples
            elif args.train_mode in ['train_lesion']:
                epoch_loss = metrics_lesion['dice'] / epoch_samples
            else:
                epoch_loss = 0


            with open(csv_path, 'a') as f:
                f.write('{},{},{},{},{},{},{},{}\n'.format(epoch, phase, metrics_liver['dice']/epoch_samples,metrics_liver['loss']/epoch_samples,metrics_liver['bce']/epoch_samples,
                                               metrics_lesion['dice']/epoch_samples,metrics_lesion['loss']/epoch_samples,metrics_lesion['bce']/epoch_samples))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss and final_eval == False:

                best_loss = epoch_loss
                print("saving best model, best_loss: ", best_loss)
                torch.save({'state_dict': model_livel.state_dict(), 'epoch': epoch, 'optim': optimizer}, checkpoint_liver_path)
                torch.save({'state_dict': model_lesion.state_dict(), 'epoch': epoch, 'optim': optimizer}, checkpoint_lesion_path)

                torch.save({'state_dict': model_livel.state_dict()}, checkpoint_liver_path)
                torch.save({'state_dict': model_lesion.state_dict()}, checkpoint_lesion_path)
            elif(phase == 'val'):
                print("curr result: " , epoch_loss , "best result: " , best_loss)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 2

#extra_channels_first_network = 3
#model_livel = unet.UNet(in_cannels=3, n_class=extra_channels_first_network+num_class).to(device)
#model_lesion = unet.UNet(in_cannels=extra_channels_first_network+5, n_class=num_class).to(device)

# model_livel = R2AttU_Net(img_ch=3,output_ch=2).to(device)
# model_lesion = R2AttU_Net(img_ch=5,output_ch=2).to(device)


model_livel = AttU_Net(img_ch=9,output_ch=3).to(device)
if args.train_mode in ['train_lesion' , 'test']:
    #model_lesion = unet.UNet(in_cannels=11, n_class=num_class).to(device)
    model_lesion = AttU_Net(img_ch=9, output_ch=3).to(device)
else:
    model_lesion = model_livel #kind of patch
#model_lesion = AttU_Net(img_ch=5,output_ch=2).to(device)

if args.train_mode in ['train_lesion' , 'test']:
    checkpoint_liver = torch.load(checkpoint_liver_path)
    model_livel.load_state_dict(checkpoint_liver['state_dict'], strict=True)

    if args.train_mode == 'test':
        checkpoint_lesion = torch.load(checkpoint_lesion_path)
        model_lesion.load_state_dict(checkpoint_lesion['state_dict'],strict=True)

# Observe that all parameters are being optimized
if args.train_mode in ['train_lesion' , 'test']:
    optimizer_ft = optim.Adam(itertools.chain(model_lesion.parameters(), model_livel.parameters()), lr=1e-4)
else:
    optimizer_ft = optim.Adam(itertools.chain( model_livel.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1) #TODO


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
if args.train_mode in ['train_lesion' , 'train_liver']:
    required_phanes = ['train', 'val']
    if (args.start_epoch< args.epochs):
        train_model(model_livel,model_lesion, optimizer_ft, exp_lr_scheduler, dataloaders , required_phanes ,num_epochs=args.epochs - args.start_epoch)
else:
    required_phanes = ['val']
    train_model(model_livel,model_lesion, optimizer_ft, exp_lr_scheduler, dataloaders ,required_phanes, num_epochs=1,final_eval = True)


