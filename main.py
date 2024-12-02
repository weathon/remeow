# %%




# %%
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image
# from video_model import MyModel
# from mae_encoder import MyModel  
# from matching_model import MyModel
# from dual_stream import MyModel
# from better_backbone import MyModel
# from better_backbone_with_3d_conv import MyModel
# from simple_3dconv import MyModel
from conv3d_with_refine import MyModel
# from cross_attn import MyModel
# from flow import MyModel
# from matching_model import MyModel
# from hand_attention import MyModel
# from seg_unet import MyModel
# from simple_conv import MyModel
from trainer import Trainer
# from is_net_backbone import ISNetBackbone

from video_dataloader import CustomDataset
# from dataloader import CustomDataset
import wandb 

import os
import argparse


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('--fold', type=int, required=True, help='Fold number for cross-validation')
parser.add_argument('--gpu', type=str, default="0", help='GPU id to use')
parser.add_argument('--refine_mode', type=str, default="residual", help='Refine mode', choices=["residual", "direct"])
parser.add_argument('--noise_level', type=float, default=1, help='Noise level') 
parser.add_argument('--steps', type=int, default=25000, help='Number of steps to train')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate') 
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
parser.add_argument('--mask_upsample', type=str, default="interpolate", help='Mask upsample method', choices=["interpolate", "transpose_conv", "shuffle"])
parser.add_argument('--refine_see_bg', action="store_true", help='If refine operator can see background')
parser.add_argument('--backbone', type=str, default="4", help='Backbone size to use', choices=["0", "1", "2", "3", "4"])
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu






model = MyModel(args) 
# model = ISNetBackbone(args) 
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps) 
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20, verbose=True, cooldown=5, threshold=0.001) 
# train_dataset = CustomDataset("/home/wg25r/preaug_cdnet/", "/home/wg25r/CDNet", 4, "train")
# val_dataset = CustomDataset("/home/wg25r/preaug_cdnet/", "/home/wg25r/CDNet", 4, "val")
train_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args.fold, "train")
val_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args.fold, "val")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)

def iou_loss(pred, target, ROI): 
    pshape = pred.shape[:1] + pred.shape[2:]
    assert pred.shape[1] == 5, f"pred shape: {pred.shape}"

    assert pshape == target.shape == ROI.shape, f"pred shape: {pshape}, target shape: {target.shape}, ROI shape: {ROI.shape}"
    # print(torch.sigmoid(pred).max(), ROI.max(), target.max())
    # pred = torch.sigmoid(pred)[ROI>0.9] hu xi 
    
    total_loss = 0
    for i in range(5):
        pred_ = pred[:,i][ROI>0.9]
        target_ = target.float()[ROI>0.9] 
        intersection = (pred_ * target_).sum()
        union = pred_.sum() + target_.sum() - intersection 
        iou = (intersection + 1e-6) / (union + 1e-6)
        total_loss +=  (1 - iou) * (0.6 ** (5 - i - 1))
        
    return total_loss


def matching_loss(pred, target, ROI):
    assert pred.shape == target.shape == ROI.shape, f"pred shape: {pred.shape}, target shape: {target.shape}, ROI shape: {ROI.shape}"
    assert len(pred.shape) == 4
    assert pred.shape[1:] == (640, 640, 256)
    
    pred = pred[ROI>0.9]
    target = target[ROI>0.9]
    assert pred.shape[2:]==(256,)
    matching_volume = torch.einsum("ble,ble->bll", pred, pred)
    axis1 = pred.reshape(pred.shape[0], -1)
    
    matching_volume = torch.sigmoid(matching_volume) 
    
    
loss_fn = iou_loss

wandb.init(project="Remeow", config=args)
wandb.define_metric("pstep")
logger = wandb
# model = torch.nn.DataParallel(model).cuda()
model = model.cuda()
# model.load_state_dict(torch.load("model.pth"))
trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn, args)


# %%
trainer.train()


