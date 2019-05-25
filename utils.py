
import random
import torch
import numpy as np
from PIL import Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_images(img, file_name):
    im = tensor2im(img)
    image_name = file_name # '%s_%s.png' % (name, label)
    if not os.path.isdir('result'):
        os.makedirs('result')
    save_path = os.path.join('result', image_name)
    save_image(im, save_path)

def save_seg_images(img, map, lesion_map, file_name):
    # im = tensor2im(img)
    image_name = file_name # '%s_%s.png' % (name, label)
    if not os.path.isdir('./result/overlay'):
        os.makedirs('./result/overlay')
    save_path = os.path.join('./result/overlay', image_name)
    map = torch.unsqueeze(map, 2).repeat(1,1,4)
    map[:,:,1:2] = 0
    lesion_map = torch.unsqueeze(lesion_map, 2).repeat(1,1,4)
    lesion_map[:,:,0] = 0
    lesion_map[:,:,2] = 0


    plt.figure()
    plt.imshow(((img+1)/2).permute(1,2,0).cpu(), interpolation='none')
    plt.imshow(map.cpu())
    plt.imshow(lesion_map.cpu())
    plt.savefig(save_path)
    plt.clf()

class JointHorizontalFlip(object):
    """Horizontally flip the given pair of PIL Images randomly with a probability of 0.5."""

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.
        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        if random.random() < 0.5:
            return img[:,:,::-1].copy(), target[:,:,::-1].copy() #F.hflip(img), F.hflip(target)
        return img, target

class JointVerticalFlip(object):
    """Vertically flip the given pair of PIL Images randomly with a probability of 0.5."""

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.
        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        if random.random() < 0.5:
            return img[:,::-1,:].copy(), target[:,::-1,:].copy() # F.vflip(img), F.vflip(target)
        return img, target


class TensorJointHorizontalFlip(object):
    """Horizontally flip the given pair of PIL Images randomly with a probability of 0.5."""

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.

        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        if random.random() < 0.5:
            return img.flip(2).clone() ,  target.flip(2).clone() # img[:,:,::-1].copy(), target[:,:,::-1].copy() #F.hflip(img), F.hflip(target)
        return img, target


class TensorJointRotate(object):
    """Horizontally flip the given pair of PIL Images randomly with a probability of 0.5."""

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.

        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        random_value = random.random()
        if random_value < 0.25: #no change
            return img, target
        elif random_value < 0.5: #90 degress
            return img.clone().transpose(1, 2)  , target.clone().transpose(1, 2)
        elif random_value < 0.75: # 180 degress
            return img.flip(1).clone(), target.flip(1).clone()
        else: #270 degress
            return img.clone().transpose(1, 2).flip(2)  , target.clone().transpose(1, 2).flip(2)

class JointNormailze(object):
    """Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]"""
    def __init__(self, means, stds):
        self.means, self.stds = means, stds


    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.
        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        img -= np.array(self.means)[:,None,None]
        img /= np.array(self.stds)[:,None,None]

        return img, target

class JointCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class JointToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic1, pic2):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic1), to_tensor(pic2)

def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """

    # handle numpy array
    img = torch.from_numpy(pic)
    # HACK
    return img.float()