import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1               = np.random.randint(0, 4)
        sample['LR']     = np.rot90(sample['LR'], k1).copy()
        sample['HR']     = np.rot90(sample['HR'], k1).copy()
        k2               = np.random.randint(0, 4)
        sample['Ref']    = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample

class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR']     = np.fliplr(sample['LR']).copy()
            sample['HR']     = np.fliplr(sample['HR']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref']    = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR']     = np.flipud(sample['LR']).copy()
            sample['HR']     = np.flipud(sample['HR']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref']    = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample

class ToTensor(object):
    def __call__(self, sample):
        LR, HR, Ref, Ref_sr = sample['LR'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR     = LR.transpose((2,0,1))
        HR     = HR.transpose((2,0,1))
        Ref    = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}

class TrainSet(Dataset):
    def __init__(self, args, transform = transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        self.args = args
        self.input_list = [os.path.join(args.dataset_dir, name.split('/')[-1]) for name in 
            args.dataset_training]
        
        # Take images from generated vae
        self.lr_list = [os.path.join(args.dataset_lr_dir,  name.split('/')[-1]) for name in 
            args.dataset_training]
        
        # Take a images from original dataset
        self.ref_list   = [os.path.join(args.dataset_dir, name.split('/')[-1]) for name in 
            args.dataset_training_ref]
        
        if args.ref_as_refsr :
            self.refup_list = self.ref_list
        # Take images from pix2pix
        else : 
            self.refup_list = [os.path.join(args.dataset_ref_dir, name.split('/')[-1]) for name in 
            self.ref_list]
        
        #self.refup_list = [os.path.join(args.dataset_ref_dir, name.split('/')[-1]) for name in self.ref_list]
                        
        self.transform  = transform
        
    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # HR
        HR  = imread(self.input_list[idx])
        HR  = np.array(Image.fromarray(HR).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))

        h,w = HR.shape[:2]
        # LR and LR_sr
        LR  = imread(self.lr_list[idx])
        LR   = np.array(Image.fromarray(LR).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))
        # Ref and Ref_sr
        Ref    = imread(self.ref_list[idx])
        Ref    = np.array(Image.fromarray(Ref).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))
        
        Ref_sr = imread(self.refup_list[idx])
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))

        # change type
        LR     = LR.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # rgb range to [-1, 1]
        LR     = LR / 127.5 - 1.
        HR     = HR / 127.5 - 1.
        Ref    = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR'     : LR,  
                  'HR'     : HR,
                  'Ref'    : Ref, 
                  'Ref_sr' : Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return self.input_list[idx], sample

class TestSet(Dataset):
    def __init__(self, args, transform = transforms.Compose([ToTensor()]) ):
        self.args = args
        self.input_list = [os.path.join(args.dataset_dir, name.split('/')[-1]) for name in 
            args.dataset_testing]
        
        # Take images from generated vae
        self.lr_list = [os.path.join(args.dataset_lr_dir, name.split('/')[-1]) for name in 
            args.dataset_testing]

        # Take a images from original dataset
        self.ref_list   = [os.path.join(args.dataset_dir, name.split('/')[-1]) for name in 
            args.dataset_testing_ref]
        
        if args.ref_as_refsr :
            self.refup_list = self.ref_list
        # Take images from pix2pix
        else : 
            self.refup_list = [os.path.join(args.dataset_ref_dir, name.split('/')[-1]) for name in 
            self.ref_list]


        self.transform  = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # HR
        HR  = imread(self.input_list[idx])
        HR  = np.array(Image.fromarray(HR).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))

        h,w = HR.shape[:2]
        # LR and LR_sr
        LR  = imread(self.lr_list[idx])
        LR   = np.array(Image.fromarray(LR).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))
        # Ref and Ref_sr
        Ref    = imread(self.ref_list[idx])
        Ref    = np.array(Image.fromarray(Ref).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))
        
        Ref_sr = imread(self.refup_list[idx])
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((self.args.img_size, self.args.img_size), Image.BICUBIC))

        # change type
        LR     = LR.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # rgb range to [-1, 1]
        LR     = LR / 127.5 - 1.
        HR     = HR / 127.5 - 1.
        Ref    = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR'     : LR,  
                  'HR'     : HR,
                  'Ref'    : Ref, 
                  'Ref_sr' : Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return self.input_list[idx], sample
