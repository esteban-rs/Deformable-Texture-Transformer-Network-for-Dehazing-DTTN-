import math
import numpy as np
import logging
import cv2
import os
import shutil

import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Save results
def prepare(sample_batched, device):
    for key in sample_batched.keys():
        sample_batched[key] = sample_batched[key].to(device)
    return sample_batched

def plot_results(model = None, dataloader = None, total_images = 5, save = False, epoch = 0, device = None, args = None) :
    i = 0
    for names, sample_batched in dataloader['test'] :
        sample_batched = prepare(sample_batched, device)
        lr             = sample_batched['LR']
        hr             = sample_batched['HR']
        ref            = sample_batched['Ref']
        ref_sr         = sample_batched['Ref_sr']
    
        print(names)

        model.eval()
        with torch.no_grad():
            sr, S, _, _, _ = model(lr = lr, ref = ref, refsr = ref_sr)
            sr_save = (sr+1.) * 127.5
            
            lr     = (lr     + 1) * 127.5
            hr     = (hr     + 1) * 127.5
            ref    = (ref    + 1) * 127.5
            ref_sr = (ref_sr + 1) * 127.5
            
            sr_save_ = np.transpose(sr_save[0].squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            lr_      = np.transpose(lr[0].squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            hr_      = np.transpose(hr[0].squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            ref_     = np.transpose(ref[0].squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            ref_sr_  = np.transpose(ref_sr[0].squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

            # plot
            
            '''
            names       = ['LR', 'Extracter', 'Ref', 'Ref_sr', 'Target']
            num_classes = len(names)

            fig, ax = plt.subplots(figsize = (50,20), nrows = 1, ncols = 5, sharex = True, sharey = True,)
            ax      = ax.flatten()

            ax[0].imshow(lr_)
            ax[0].set_xlabel(str(0) + ': '+ names[0])


            ax[1].imshow(sr_save_)
            ax[1].set_xlabel(str(1) + ': '+ names[1])
            
            ax[2].imshow(ref_)
            ax[2].set_xlabel(str(2) + ': '+ names[2])
            
            ax[3].imshow(ref_sr_)
            ax[3].set_xlabel(str(3) + ': '+ names[3])
            
            ax[4].imshow(hr_)
            ax[4].set_xlabel(str(4) + ': '+ names[4])


            ax[0].set_xticks([])
            ax[0].set_yticks([])
            plt.show()

            if save :
                # Save the full figure...
                fig.savefig(args.save_dir + '/model_' + str(epoch) + '_' + str(i) + '.png')
                
            map_ = sns.heatmap(S[0].detach().cpu().numpy()[0][0], linewidth=0)
            plt.show()
            if save :
                figure = map_.get_figure()    
                figure.savefig(args.save_dir + '/heatmap_' + str(epoch) + '_' + str(i) + '.png')
        
        if i == total_images - 1 : 
            return
        i += 1
        '''
        
            names       = ['LR', 'Extracter', 'Ref', 'Ref_sr', 'Target']
            num_classes = len(names)

            fig, ax = plt.subplots(figsize = (50,20), nrows = 2, ncols = 3)

            ax[0][0].imshow(lr_)
            ax[0][0].set_xlabel(str(0) + ': '+ names[0])


            ax[0][1].imshow(sr_save_)
            ax[0][1].set_xlabel(str(1) + ': '+ names[1])
            
            ax[0][2].imshow(hr_)
            ax[0][2].set_xlabel(str(4) + ': '+ names[4])
            
            ax[1][0].imshow(ref_sr_)
            ax[1][0].set_xlabel(str(3) + ': '+ names[3])
            
            ax[1][1].imshow(ref_)
            ax[1][1].set_xlabel(str(2) + ': '+ names[2])
            
            map_ = S[0].detach().cpu().numpy()[0][0]

            ax[1][2].imshow(map_)
            ax[1][2].set_xlabel('Heat Map')


            plt.show()

            if save :
                # Save the full figure...
                fig.savefig(args.save_dir + '/model_' + str(epoch) + '_' + str(i) + '.png')
        
        if i == total_images - 1 : 
            return
        i += 1