import os
import json
from sre_parse import OCTDIGITS
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.utils.data as data

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class ADE20KSegmentation(data.Dataset):
    cmap = voc_cmap()
    def __init__(self, root, image_set='train', transform=None, dram_class=False):

        self.root = os.path.expanduser(root)
        self.ade20k_path = "ade20k"
        self.transform = transform
        self.image_set = image_set
        self.odgt_name = ""
        self.dram_class = dram_class

        if image_set == 'train':
            self.odgt_name = "training.odgt"
        else:
            self.odgt_name = "validation.odgt"
        self.root_ade20k = os.path.join(self.root, self.ade20k_path)
        self.odgt = os.path.join(self.root_ade20k, self.odgt_name)

        self.list_sample = []
        self.num_samle = 0
        self.images = []
        self.masks = []
        
        self.parse_input_list(self.odgt)
        
        self._get_img_list()

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_img_list(self):
        for idx in range(self.num_sample):
            self.images.append(os.path.join(self.root_ade20k, self.list_sample[idx]['fpath_img']))
        for idx in range(self.num_sample):
            self.masks.append(os.path.join(self.root_ade20k, self.list_sample[idx]['fpath_segm']))
        print(self.images[1])
        #print(self.images)

    def class_changer(self, mask):
        num_mask = np.array(mask)
        # changed wall 1 <- 9,15,33,43,44,145
        np.place(num_mask, ((num_mask == 9) | (num_mask == 15) | (num_mask == 33) | (num_mask == 43) | (num_mask == 44) | (num_mask == 145) ), 1)
        # changed floor 4 <- 7,14,30,53,55
        np.place(num_mask, ((num_mask == 7) | (num_mask == 14) | (num_mask == 30) | (num_mask == 53) | (num_mask == 55)), 4)
        # changed tree 5 <- 8,11,14,16,19,20,25,34
        np.place(num_mask, (num_mask == 18), 5) 
        # changed furniture 8 <- 8,11,14,16,19,20,25,34
        np.place(num_mask, ((num_mask == 11) | (num_mask == 14) | (num_mask == 16) | (num_mask == 19) | (num_mask == 20) | (num_mask == 25) | (num_mask == 34)), 8)
        # changed stairs 7 <- 54
        np.place(num_mask, (num_mask == 54), 7)
        # changed other 26
        np.place(num_mask, ((num_mask != 0) & (num_mask != 1) & (num_mask != 4) & (num_mask != 5) & (num_mask != 7) & (num_mask != 8)), 26)
        
        pil_mask = Image.fromarray(num_mask)
        
        return pil_mask

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.dram_class is True:
            target = self.class_changer(target)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]