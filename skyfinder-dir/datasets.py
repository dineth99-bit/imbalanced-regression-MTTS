import logging
import os

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window

print = logging.info


class SkyFinder(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2,
                 label_min=-40, label_max=50, crop_mode='full', mask_dir=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.crop_mode = crop_mode
        self.mask_dir = mask_dir
        self.label_min = label_min
        self.label_max = label_max
        self.num_bins = label_max - label_min + 1

        self.weights = self._prepare_weights(
            reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.df)

    def _bin_index(self, label):
        return min(self.num_bins - 1, max(0, int(round(label)) - self.label_min))

    def _apply_mask_crop(self, img, camera_id):
        if self.crop_mode == 'full' or not self.mask_dir:
            return img
        mask_path = os.path.join(self.mask_dir, 'skyfinder_masks', f'{int(camera_id)}.png')
        if not os.path.isfile(mask_path):
            return img
        mask = Image.open(mask_path).convert('L').resize(img.size)
        if self.crop_mode == 'sky_bbox':
            bbox = mask.getbbox()
            if bbox:
                img = img.crop(bbox)
        elif self.crop_mode == 'sky_masked':
            img = img.copy()
            img.putalpha(mask)
            bg = Image.new('RGB', img.size, (0, 0, 0))
            bg.paste(img, mask=mask)
            img = bg
        return img

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        img = self._apply_mask_crop(img, row['camera_id'])
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['temperature']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        return img, label, weight

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def _prepare_weights(self, reweight, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

        value_dict = {x: 0 for x in range(self.num_bins)}
        labels = self.df['temperature'].values
        for label in labels:
            value_dict[self._bin_index(label)] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}
        num_per_label = [value_dict[self._bin_index(label)] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[self._bin_index(label)] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
