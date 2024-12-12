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
# from conv3d_with_refine import MyModel
# from cross_attn import MyModel
# from flow import MyModel
# from matching_model import MyModel
# from hand_attention import MyModel
# from seg_unet import MyModel
# from simple_conv import MyModel
from trainer import Trainer
# from is_net_backbone import ISNetBackbone

# from video_dataloader import CustomDataset

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
parser.add_argument('--backbone', type=str, default="4", help='Backbone size to use', choices=["0", "1", "2", "3", "4", "5"])
parser.add_argument('--refine_steps', type=int, default=5, help='Number of refine steps')
parser.add_argument('--background_type', type=str, default="mog2", choices=["mog2", "sub"], help='Background type, mog2 means MOG2, sub means SuBSENSE')
parser.add_argument('--histogram', action="store_true", help='If use histogram')
parser.add_argument('--clip', type=float, default=1, help='Gradient clip norm')
parser.add_argument('--note', type=str, default="", help='Note for this run (for logging purpose)')
parser.add_argument('--conf_penalty', type=float, default=0, help='Confidence penalty, penalize the model if it is too confident')
parser.add_argument('--image_size', type=int, default=512, help="Image size", choices=[512, 640])
parser.add_argument('--hard_shadow', action="store_true", help='If use hard shadow')
parser.add_argument('--lambda2', type=float, default=30, help='Lambda2 for pretrained weights and new weights')
parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum learning rate')
parser.add_argument('--print_every', type=int, default=100, help='Print every n steps')
parser.add_argument('--val_size', type=int, default=1024, help='Validation size')

args = parser.parse_args()

if args.image_size == 512:
    assert args.backbone in ["0", "1", "2", "3", "4"], "Backbone size should be 0, 1, 2, 3, 4 when image size is 512"
elif args.image_size == 640:
    assert args.backbone == "5", "Backbone size should be 5 when image size is 640"
    
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from histgram_3dconv_norefine import MyModel
from video_histgram_dataloader import CustomDataset



model = MyModel(args) 
# model = ISNetBackbone(args) 
model = torch.nn.DataParallel(model).cuda() #should be here before optimizer, that was why no converge and error
# head conv3d t_dim upsample hist_encoder
# optimizer = torch.optim.AdamW(
#     [
#         {'params': model.module.head.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
#         {'params': model.module.conv3d.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
#         {'params': model.module.t_dim.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
#         {'params': model.module.upsample.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
#         {'params': model.module.hist_encoder.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
#         {'params': model.module.backbone.parameters(), 'lr': args.learning_rate * 0.2, "weight_decay": args.weight_decay * 0.2},
#     ]
#     , lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr_min)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20, verbose=True, cooldown=5, threshold=0.001) 
# train_dataset = CustomDataset("/home/wg25r/preaug_cdnet/", "/home/wg25r/CDNet", 4, "train")
# val_dataset = CustomDataset("/home/wg25r/preaug_cdnet/", "/home/wg25r/CDNet", 4, "val")
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

def mutli_class_iou_loss(pred, target, ROI):
    assert pred.shape[1] == 3, f"pred shape: {pred.shape}"
    total_loss = 0
    for class_name in range(3):
        pred_ = pred[:,class_name][ROI>0.9]
        target_ = target.float()[ROI>0.9] == class_name
        intersection = (pred_ * target_).sum()
        union = pred_.sum() + target_.sum() - intersection 
        iou = (intersection + 1e-6) / (union + 1e-6)
        conf = (pred_ - 0.5).abs().mean()
        conf_pen = conf * args.conf_penalty
        total_loss +=  (1 - iou + conf_pen)
    return total_loss


def regularization_loss(model_0, model_t):
    total_loss = 0
    count = 0
    for param_0, param_t in zip(model_0, model_t):
        total_loss += (param_0 - param_t).abs().sum()
        count += param_0.size()[0]
        
    return total_loss / count
    
loss_fn = iou_loss if not args.hard_shadow else mutli_class_iou_loss

wandb.init(project="Remeow", config=args)
wandb.define_metric("pstep")
logger = wandb
# model = model.cuda()
# model.load_state_dict(torch.load("model.pth"))
trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn, args, regularization_loss)


# %%
trainer.train()


