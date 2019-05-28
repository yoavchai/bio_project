import torch
import torch.nn as nn


def dice_loss(pred, target, metrics, smooth=1.,binary=False):
    pred = pred.contiguous()
    target = target.contiguous()

    #pred[:,1]  = (pred[:,1] > pred[:,0]).to(dtype=torch.float32)
    #pred[:,0] = (pred[:,0] >= pred[:,1]).to(dtype=torch.float32)

    # if binary == True:
    #     threshold = 0
    #
    #     #for liver
    #     # pred  = ((pred[:,1] + threshold > pred[:,0]) * (pred[:,1] + threshold > pred[:,2])).to(dtype=torch.float32)
    #     # target_for_calc = ((target[:,1] + threshold > target[:,0])*(target[:,1] + threshold > target[:,2])).to(dtype=torch.float32)
    #
    #     ##for lesion
    #     # pred  = ((pred[:,2] + threshold > pred[:,1]) * (pred[:,2] + threshold > pred[:,0])).to(dtype=torch.float32)
    #     # target_for_calc = ((target[:,2] + threshold > target[:,1])*(target[:,2] + threshold > target[:,0])).to(dtype=torch.float32)
    #
    #     intersection = (pred * target_for_calc).sum(dim=1).sum(dim=1)
    #     loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + target_for_calc.sum(dim=1).sum(dim=1) + smooth)))
    #
    # else:
    pred_argmax = torch.argmax(pred,1)

    pred_bg = (pred_argmax == 0).float()
    pred_liver = (pred_argmax == 1).float()
    pred_lesion = (pred_argmax == 2).float()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    intersection_bg = (pred_bg * target[:,0]).sum(dim=1).sum(dim=1)
    intersection_liver = (pred_liver * target[:,1]).sum(dim=1).sum(dim=1)
    intersection_lesion = (pred_lesion * target[:,2]).sum(dim=1).sum(dim=1)

    loss = (1 - ((2. * intersection + smooth) / (
                pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    loss_bg = (1 - ((2. * intersection_bg + smooth) / (
            pred_bg.sum(dim=1).sum(dim=1) + target[:,0].sum(dim=1).sum(dim=1) + smooth)))
    loss_liver = (1 - ((2. * intersection_liver + smooth) / (
            pred_liver.sum(dim=1).sum(dim=1) + target[:, 1].sum(dim=1).sum(dim=1) + smooth)))
    loss_lesion = (1 - ((2. * intersection_lesion + smooth) / (
            pred_lesion.sum(dim=1).sum(dim=1) + target[:, 2].sum(dim=1).sum(dim=1) + smooth)))

    liver_precision  = (intersection_liver / pred_liver.sum(1).sum(1))
    liver_recall     = (intersection_liver / target[:,1].sum(1).sum(1))
    lesion_precision = (intersection_lesion / pred_lesion.sum(1).sum(1))
    lesion_recall    = (intersection_lesion / target[:, 2].sum(1).sum(1))

    # loss = loss_bg + loss_liver + loss_lesion
    metrics['dice']         += loss.mean() * target.size(0)
    metrics['bg_dice']      += loss_bg.mean() * target.size(0)
    metrics['liver_dice']   += loss_liver.mean() * target.size(0)
    metrics['lesion_dice']  += loss_lesion.mean() * target.size(0)
    metrics['liver_precision']  += liver_precision.mean() * target.size(0)
    metrics['liver_recall']     += liver_recall.mean() * target.size(0)
    if not torch.isnan(lesion_precision).any():
        metrics['lesion_precision'] += lesion_precision.mean() * target.size(0)
        metrics['lesion_precision_samples'] += target.size(0)
    if not torch.isnan(lesion_recall).any():
        metrics['lesion_recall']    += lesion_recall.mean() * target.size(0)
        metrics['lesion_recall_samples'] += target.size(0)

    # loss = (1 - ((2. * intersection + smooth) / (
    #     torch.abs(pred).sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))  #TODO

    return loss.mean()