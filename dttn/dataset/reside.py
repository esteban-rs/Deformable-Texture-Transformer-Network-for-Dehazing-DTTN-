import os
from imageio import imread
from PIL import Image, ImageFile

import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, HR, Ref, Ref_sr = sample['LR'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}

def generate_reference_sample(img_name, input_list) :
    img_name_ = img_name.split('/')[-1].split('_')[0]
    
    is_the_same = True
    while is_the_same:
        ref_name = np.random.choice(input_list)
        if img_name_ != ref_name.split('/')[-1].split('_')[0] :
            is_the_same = False
    return ref_name
    
class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), ToTensor()]) ):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'train/GT', '*.png')))
        self.hazy_list  = sorted(glob.glob(os.path.join(args.dataset_dir, 'train/hazy', '*.png')))
            
        self.ref_list      = [generate_reference_sample(name, self.input_list) for name in self.input_list]
        self.ref_hazy_list = [name.replace('GT', 'hazy') for name in self.ref_list]
        
        if args.aumentation :
            self.aumentation = True
            self.transform = transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])  
        else :
            self.aumentation = False
            self.transform = transforms.Compose([ToTensor()]) 
               
        self.filled    = 4 * args.stride_size
        self.img_size  = [args.img_size[0] // self.filled * self.filled, args.img_size[1] // self.filled * self.filled]


    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # HR
        HR  = imread(self.input_list[idx])
        HR  = np.array(Image.fromarray(HR).resize(self.img_size, Image.BICUBIC))

        # LR and LR_sr
        LR  = imread(self.hazy_list[idx])
        LR  = np.array(Image.fromarray(LR).resize(self.img_size, Image.BICUBIC))
        
        
        ### Ref and Ref_sr
        # Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        Ref = np.array(Image.fromarray(Ref).resize(self.img_size, Image.BICUBIC))
            
        Ref_sr = imread(self.ref_hazy_list[idx])
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize(self.img_size, Image.BICUBIC))

        ### change type
        LR     = LR.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR     = LR / 127.5 - 1.
        HR     = HR / 127.5 - 1.
        Ref    = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,  
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return self.input_list[idx], sample

class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.args = args

        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/GT', '*.png')))
        self.hazy_list  = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/hazy', '*.png')))
        
        self.ref_list      = [generate_reference_sample(name, self.input_list) for name in self.input_list]
        self.ref_hazy_list = [name.replace('GT', 'hazy') for name in self.ref_list]

        self.transform  = transform
        # Crop image dependig on kernel size
        self.filled     = 4 * args.stride_size
        
        
        self.img_size = [self.args.img_size[0] // self.filled * self.filled, self.args.img_size[1] // self.filled * self.filled]

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # HR
        HR  = imread(self.input_list[idx])
        HR  = np.array(Image.fromarray(HR).resize(self.img_size, Image.BICUBIC))

        # LR and LR_sr
        LR  = imread(self.hazy_list[idx])
        LR  = np.array(Image.fromarray(LR).resize(self.img_size, Image.BICUBIC))
        
        
        ### Ref and Ref_sr
        # Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        Ref = np.array(Image.fromarray(Ref).resize(self.img_size, Image.BICUBIC))
            
        Ref_sr = imread(self.ref_hazy_list[idx])
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize(self.img_size, Image.BICUBIC))

        ### change type
        LR     = LR.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR     = LR / 127.5 - 1.
        HR     = HR / 127.5 - 1.
        Ref    = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,  
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return self.input_list[idx], sample
    