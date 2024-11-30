# %%
class Args:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.05
        self.hidden_dropout_prob = 0.05
        self.drop_path_rate = 0.05
        self.classifier_dropout = 0
        self.ksteps = 200 

args = Args()

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
from trainer import trainer
from is_net_backbone import ISNetBackbone

from video_dataloader import CustomDataset
# from dataloader import CustomDataset
import wandb 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

model = MyModel(args) 
# model = ISNetBackbone(args) 
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.25e-2) 

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ksteps) 
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20, verbose=True, cooldown=5, threshold=0.001) 
# train_dataset = CustomDataset("/mnt/fastdata/preaug_cdnet/", "/mnt/fastdata/CDNet", 4, "train")
# val_dataset = CustomDataset("/mnt/fastdata/preaug_cdnet/", "/mnt/fastdata/CDNet", 4, "val")
train_dataset = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 2, "train")
val_dataset = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 2, "val")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=70, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=70, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)

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
        total_loss +=  (1 - iou) * (0.8 ** (5-i))
        
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

wandb.init(project="Remeow")
wandb.define_metric("pstep")
logger = wandb
# model = torch.nn.DataParallel(model).cuda()
model = model.cuda()
# model.load_state_dict(torch.load("model.pth"))
trainer = trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn)


# %%
trainer.train()


