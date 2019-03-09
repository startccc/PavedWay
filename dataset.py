import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms
import torchvision.transforms.functional as TF
import os
from skimage import io, transform
from PIL import Image
import numpy as np
import random

class SateliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.randomCrop = transforms.CenterCrop(400)  # RandomCrop(400)
        self.idx2path = dict(enumerate(os.listdir(os.path.join(self.root_dir, 'images/'))))

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, 'images/')))

    def __getitem__(self, idx):
        img_dir = os.path.join(self.root_dir, 'images/')
        gt_dir = os.path.join(self.root_dir, 'groundtruth/')
        img_path = os.path.join(img_dir, self.idx2path[idx])
        gt_path = os.path.join(gt_dir, self.idx2path[idx])
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        gt = Image.open(gt_path).convert('L')
        sample = {'image': image, 'gt': gt}
        
        # random rotation
        angle = np.random.choice([0, 90, -90, 45, -45, 180])
        sample['image'] = TF.rotate(sample['image'], angle)
        sample['gt'] = TF.rotate(sample['gt'], angle)

        if random.random() > 0.5:
            sample['image'] = TF.hflip(sample['image'])
            sample['gt'] = TF.hflip(sample['gt'])

        # Random vertical flipping
        if random.random() > 0.5:
            sample['image'] = TF.vflip(sample['image'])
            sample['gt'] = TF.vflip(sample['gt'])
        
        if self.transform:
            sample['image'] = self.randomCrop(sample['image'])
            sample['gt'] = self.randomCrop(sample['gt'])
            sample = self.transform(sample)
            # sample = self.randomCrop(sample)

        return sample

class ToTensor2(object):
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(np.array(sample['image']) / 255.0).to(dtype=torch.float32).permute(2, 0, 1)
        sample['gt'] = torch.from_numpy((np.array(sample['gt']) > 127).astype(np.uint8)).to(dtype=torch.float32)
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['gt']
        image = image.permute(1,2,0)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        landmarks = landmarks[top: top + new_h,
                      left: left + new_w]
        image = image.permute(2,0,1)

        return {'image': image, 'gt': landmarks}