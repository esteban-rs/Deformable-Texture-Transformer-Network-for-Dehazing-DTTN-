import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from model.Modules import conv3x3, ResBlock
# from model.DCN import DeformableConv2d

class DeformableAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(DeformableAlignment, self).__init__()
        self.padding     = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size = kernel_size, 
                                     stride      = stride,
                                     padding     = self.padding, 
                                     bias        = True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size * kernel_size,
                                     kernel_size = kernel_size, 
                                     stride      = stride,
                                     padding     = self.padding, 
                                     bias        = True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels  = in_channels - out_channels,
                                      out_channels = out_channels,
                                      kernel_size  = kernel_size,
                                      stride       = stride,
                                      padding      = self.padding,
                                      bias         = True)
    
    def forward(self, x, ref_feats):
        b, c, h, w = x.shape
        max_offset = max(h, w)/4.
        
        x_cat     = torch.cat([x, ref_feats], dim = 1)
        offset    = self.offset_conv(x_cat).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x_cat))
        
        ref_feats = torchvision.ops.deform_conv2d(input   = ref_feats, 
                                                  offset  = offset, 
                                                  weight  = self.regular_conv.weight, 
                                                  bias    = self.regular_conv.bias, 
                                                  padding = self.padding,
                                                  mask    = modulator)
        return ref_feats
    
class DeformableFusion(nn.Module):
    def __init__(self, init_n_feats, n_feats, top_k,  num_res_blocks):
        super(DeformableFusion, self).__init__()
        self.num_res_blocks = num_res_blocks

        self.conv_head = nn.ModuleList()
        for i in range(top_k):
            self.conv_head.append(DeformableAlignment(init_n_feats + n_feats, n_feats))
        
        self.fuse     = conv3x3(n_feats + n_feats, n_feats)

        self.RB       = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB.append(ResBlock(in_channels  = n_feats,  middle_channels = n_feats, out_channels = n_feats))

    def forward(self, x, T, S):
        attention = 0 
        for i in range(len(T)) :
            S_        =  F.interpolate(S[i], size = (x.shape[2], x.shape[3]), mode='bicubic')
            f_deform  = self.conv_head[i](x, T[i] * S_) 
            attention = attention + f_deform 

        x_cat = torch.cat([x, attention], dim = 1)
        x = x + self.fuse(x_cat)

        for i in range(self.num_res_blocks):
            x = self.RB[i](x)

        
        return x
