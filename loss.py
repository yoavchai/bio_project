import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1.,binary=False):
    pred = pred.contiguous()
    target = target.contiguous()

    #pred[:,1]  = (pred[:,1] > pred[:,0]).to(dtype=torch.float32)
    #pred[:,0] = (pred[:,0] >= pred[:,1]).to(dtype=torch.float32)

    if binary == True:
        threshold = -0.1
        pred  = (pred[:,1] + threshold > pred[:,0]).to(dtype=torch.float32) #TODO maybe it is better to set a diff threhold
        target_for_calc = (target[:,1] + threshold > target[:,0]).to(dtype=torch.float32)
        intersection = (pred * target_for_calc).sum(dim=1).sum(dim=1)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + target_for_calc.sum(dim=1).sum(dim=1) + smooth)))

    else:
        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()