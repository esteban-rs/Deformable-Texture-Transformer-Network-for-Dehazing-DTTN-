import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model import Gradient
from model.Modules import conv1x1, conv3x3, ResBlock, SFE, MergeTail
from model.DF import DeformableFusion
    

class MSDI(nn.Module):
    def __init__(self, num_res_blocks, num_grad_blocks, feats_by_level, n_feats, res_scale, top_k, device):
        super(MSDI, self).__init__()
        self.num_res_blocks   = num_res_blocks 
        self.num_res_blocks_g = num_grad_blocks
        self.n_feats          = n_feats
        self.top_k            = top_k
        self.feats_by_level   = feats_by_level
        
        # FEATURE INTEGRATION
        self.SFE        = SFE(self.num_res_blocks[0], n_feats, res_scale)
        ### Level3
        self.Fusion1    = DeformableFusion(self.feats_by_level[2], n_feats, top_k, num_res_blocks[1])
        ### upsampling3
        self.conv12     = conv3x3(n_feats, n_feats*4)
        self.ps12       = nn.PixelShuffle(2)
        ### Level2
        self.Fusion2    = DeformableFusion(self.feats_by_level[1], n_feats, top_k, num_res_blocks[2])
        ### upsampling2
        self.conv23     = conv3x3(n_feats, n_feats*4)
        self.ps23       = nn.PixelShuffle(2)
        ### Level1
        self.Fusion3    = DeformableFusion(self.feats_by_level[0], n_feats, top_k, num_res_blocks[3])
        self.merge_tail = MergeTail(n_feats)

        # GRADIENT FEATURES 
        self.gradient    = Gradient.gradient(device)
        self.SFE_GRAD    = SFE(self.num_res_blocks_g[0], n_feats, res_scale)
        # Level3
        self.conv12_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_12     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[1]):
            self.grad_12.append(ResBlock(in_channels = n_feats, middle_channels = n_feats, out_channels = n_feats))
        # Level2
        self.conv23_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_23     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[2]):
            self.grad_23.append(ResBlock(in_channels = n_feats, middle_channels = n_feats, out_channels = n_feats))
        #Level1
        self.conv33_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_33     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[3]):
            self.grad_33.append(ResBlock(in_channels = n_feats, middle_channels = n_feats, out_channels = n_feats))
        # Fuse between SR and grad
        self.fuse        = conv3x3(2 * n_feats, 3)
        

    def forward(self, x, S = None, T_lv3 = None, T_lv2 = None, T_lv1 = None):
        x_sr  = x.clone()
        g     = self.gradient((x + 1) / 2)
        x     = self.SFE(x)
        
        # Feature Merging lv1
        T1    = self.Fusion1(x, T_lv3, S)                                     # [9,   64, 40, 40]
        T1_up = self.conv12(T1)                                               # [9, 64*4, 40, 40]
        T1_up = F.relu(self.ps12(T1_up))                                      # [9,   64, 80, 80]
    
        # Feature Merging lv2
        T2    = self.Fusion2(T1_up, T_lv2, S)                                 # [9,   64,  80,  80]
        T2_up = self.conv23(T2)                                               # [9, 64*4,  80,  80]
        T2_up = F.relu(self.ps23(T2_up))                                      # [9,   64, 160, 160]
        
        # Feature Merging lv3
        T3    = self.Fusion3(T2_up, T_lv1, S)                                # [9, 64, 160, 160]
        x_tt  = self.merge_tail(x, T1, T2, T3)                               # [9, 64, 160, 160]

        # Sallow Feature Extraction
        grad  = self.SFE_GRAD(g)                                             # [9,   64, 40, 40]
        x_grad1 = torch.cat([grad, T1], dim = 1)                             # [9, 64*2, 40, 40]
        x_grad1 = self.conv12_grad(x_grad1)                                  # [9,   64, 40, 40]

        for i in range(self.num_res_blocks_g[1]):
            x_grad1 = self.grad_12[i](x_grad1)
        
        x_grad2 = F.interpolate(x_grad1, scale_factor = 2, mode='bicubic')   # [9,   64, 80, 80]
        x_grad2 = torch.cat([x_grad2, T2], dim = 1)                          # [9, 2*64, 80, 80]
        x_grad2 = self.conv23_grad(x_grad2)                                  # [9,   64, 80, 80]

        for i in range(self.num_res_blocks_g[2]):
            x_grad2 = self.grad_23[i](x_grad2)
            
        x_grad3 = F.interpolate(x_grad2, scale_factor = 2, mode='bicubic')   # [9,   64, 160, 160]
        x_grad3 = torch.cat([x_grad3, T3], dim = 1)                          # [9, 2*64, 160, 160]
        x_grad3 = self.conv33_grad(x_grad3)                                  # [9,   64, 160, 160]

        for i in range(self.num_res_blocks_g[3]):
            x_grad3 = self.grad_33[i](x_grad3)
        
        x_cat = torch.cat([x_tt, x_grad3], dim = 1)                          # [9, 2*64, 160, 160] 
        x_cat = self.fuse(x_cat)                                             # [9,   64, 160, 160] 
 
        
        # Return skip HR image
        return torch.clamp(x_sr + x_cat, -1, 1)