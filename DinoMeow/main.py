from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image

from trainer import Trainer

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
parser.add_argument('--refine_steps', type=int, default=5, help='Number of refine steps')
parser.add_argument('--background_type', type=str, default="mog2", choices=["mog2", "sub"], help='Background type, mog2 means MOG2, sub means SuBSENSE')
parser.add_argument('--histogram', action="store_true", help='If use histogram')
parser.add_argument('--clip', type=float, default=1, help='Gradient clip norm')
parser.add_argument('--note', type=str, default="", help='Note for this run (for logging purpose)')
parser.add_argument('--conf_penalty', type=float, default=0, help='Confidence penalty, penalize the model if it is too confident')
parser.add_argument('--m', type=float, default=0.99, help='Exponential moving average decay')
parser.add_argument('--dropout', type=float, default=0.1, help="Drop for student")
parser.add_argument('--temp', type=float, default=4, help="Temp for teacher model")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


from histgram_3dconv_norefine import MyModel
from video_histgram_dataloader import CustomDataset



student = MyModel(args) 
student = torch.nn.DataParallel(student, "student").cuda() 
student.load_state_dict(torch.load("model.pth"))

teacher = MyModel(args)
teacher = torch.nn.DataParallel(teacher, "teacher").cuda()
teacher.load_state_dict(torch.load("model.pth"))

optimizer = torch.optim.AdamW(student.module.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps) 


train_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "train")
val_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "val")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True) 

def iou_loss(pred, target, ROI): 
    pshape = pred.shape[:1] + pred.shape[2:]
    assert pred.shape[1] == args.refine_steps, f"pred shape: {pred.shape}"
    assert pshape == target.shape == ROI.shape, f"pred shape: {pshape}, target shape: {target.shape}, ROI shape: {ROI.shape}"

    total_loss = 0
    for i in range(args.refine_steps):
        pred_ = pred[:,i][ROI>0.9]
        target_ = target.float()[ROI>0.9] 
        intersection = (pred_ * target_).sum()
        union = pred_.sum() + target_.sum() - intersection 
        iou = (intersection + 1e-6) / (union + 1e-6)
        conf = (pred_ - 0.5).abs().mean()
        conf_pen = conf * args.conf_penalty
        total_loss +=  (1 - iou + conf_pen) * (0.6 ** (args.refine_steps - i - 1))

    return total_loss



    
loss_fn = iou_loss

wandb.init(project="Remeow", config=args)
wandb.define_metric("pstep")
logger = wandb

trainer = Trainer(student, teacher, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn, args)

trainer.train()


