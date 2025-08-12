from Utils import psnr_and_ssim_torch
from model import Vgg19

import os
import time
import numpy as np
from imageio import imsave

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils

class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args       = args
        self.logger     = logger
        self.dataloader = dataloader
        self.model      = model
        self.loss_all   = loss_all
        self.device     = torch.device(args.device) 
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr_rate , betas = (args.beta1, args.beta2), 
                                                             eps   = args.eps)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = [1, 50], gamma = self.args.gamma)
        
        
        if args.dataset == 'urban100' :
            del(self.vgg19)
            del(self.optimizer)

        self.max_psnr       = 0.
        self.max_psnr_epoch = 0
        self.max_ssim       = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path = None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            model_state_dict_save = {}
            for k,v in torch.load(model_path, map_location=self.device).items() :
                if k.find('APFM1') > -1 :
                    k_ = k.replace('conv11_head', 'conv_head')
                    k_ = k_.replace('RB11', 'RB')
                elif k.find('APFM2') > -1 :
                    k_ = k.replace('conv22_head', 'conv_head')
                    k_ = k_.replace('RB22', 'RB')
                elif k.find('APFM3') > -1 :
                    k_ = k.replace('conv33_head', 'conv_head')
                    k_ = k_.replace('RB33', 'RB')
                else :
                    k_ = k
                model_state_dict_save[k_] = v

            model_state_dict      = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict, strict = True)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            if key != 'pad_nums' :
                sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0):
        self.model.train()
        self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))
        
        start = time.time()
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched[1])
            lr             = sample_batched['LR']
            hr             = sample_batched['HR']
            ref            = sample_batched['Ref']
            ref_sr         = sample_batched['Ref_sr']

            sr, _, _, _, _ = self.model(lr    = lr, 
                                                    ref   = ref, 
                                                    refsr = ref_sr)

            # calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) # flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss     = rec_loss
            if (is_print):
                self.logger.info('epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()))

            loss.backward()
            self.optimizer.step()
              
        self.logger.info('Training time for epoch ' + str(current_epoch) + ' : ' +  str(time.time() - start) +  ' seconds.')

        if current_epoch % self.args.save_every == 0 :
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
        
        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.png'), sr_save)
                    
                    ### calculate psnr and ssim
                    _psnr, _ssim = psnr_and_ssim_torch.calc_psnr_and_ssim(sr.detach(), hr.detach())
                    
                    sr_       = interpolate(sr)
                    hr_       = interpolate(hr)
                    fid_value = calculate_fid_given_paths(sr_, hr_ ,device, 2048)

                    psnr += _psnr
                    ssim += _ssim
                    fid  += fid_value

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                fid_ave  = fid /cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f \t FID (now): %.4f' %(psnr_ave, ssim_ave, fid_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                if (fid_ave > self.max_fid):
                    self.max_fid = fid_ave
                    self.max_fid_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d) \t FID (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch, self.max_fid, self.max_fid_epoch))
        
        
        
        
        elif (self.args.dataset == 'CELEBA-256' or self.args.dataset == 'RESIDE'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched[1])
                    lr = sample_batched['LR']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    sr, _, _, _, _ = self.model(lr=lr, ref=ref, refsr=ref_sr)
                    
                    
                    
                    ### calculate psnr and ssim
                    _psnr, _ssim = psnr_and_ssim_torch.calc_psnr_and_ssim(sr.detach(), hr.detach())


                    psnr += _psnr
                    ssim += _ssim
                
                
                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d) ' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))
        self.logger.info('Evaluation over.')
        
    def test(self, args):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' %(self.args.lr_path))
        self.logger.info('ref path:    %s' %(self.args.ref_path))

        ### LR and LR_sr
        LR = imread(self.args.lr_path)[:, :, :3]
        h1, w1 = LR.shape[:2]
        #LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))
        
        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)[:, :, :3]
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        
        print(LR.shape, Ref.shape)
        
        Ref    = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))
        
        LR_  = LR
        Ref_ = Ref
        ### change type
        LR     = LR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)
        
        
        ### rgb range to [-1, 1]
        LR     = LR / 127.5 - 1.
        Ref    = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        ### to tensor
        LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
                
        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _ = self.model(lr = LR_t, ref = Ref_t, refsr = Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                       
            save_path = os.path.join(self.args.save_dir, 'save_results', args.model_path.split('/')[-1].split('.')[0] + '_' + os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' %(save_path))

        self.logger.info('Test over.')
        
        return LR_, Ref_, sr_save
