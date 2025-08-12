from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.L1Loss()
    def forward(self, sr, hr):
        return self.loss(sr, hr)

def get_loss_dict(args, logger):
    loss = {}

    loss['rec_loss'] = ReconstructionLoss()
    return loss