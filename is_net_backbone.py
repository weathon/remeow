# %%
import sys
sys.path.append("./DIS/IS-Net")
from models import *

import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize


class ISNetBackbone(nn.Module):
    def __init__(self, args):
        super(ISNetBackbone, self).__init__()
        self.model = ISNetDIS().cuda()
        self.model.load_state_dict(torch.load("/home/wg25r/isnet-general-use.pth"))
        self.model.conv_in = nn.Conv2d(9, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))

    def forward(self, x):
        x = self.model(x)[0][0]
        return x
