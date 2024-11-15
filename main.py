# %%
class Args:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.drop_path_rate = 0.1
        self.classifier_dropout = 0.1
        self.ksteps = 200 

args = Args()

# %%
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image
from model import MyModel
from trainer import trainer
from is_net_backbone import ISNetBackbone

from dataloader import CustomDataset
import wandb 

# model = MyModel(args)
model = ISNetBackbone(args)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ksteps)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
# train_dataset = CustomDataset("/mnt/fastdata/preaug_cdnet/", "/mnt/fastdata/CDNet", 4, "train")
# val_dataset = CustomDataset("/mnt/fastdata/preaug_cdnet/", "/mnt/fastdata/CDNet", 4, "val")
train_dataset = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "train")
val_dataset = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "val")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=70, pin_memory=True, persistent_workers=True, prefetch_factor=2) 
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=70, pin_memory=True, persistent_workers=True, prefetch_factor=2) 

def iou_loss(pred, target, ROI):
    assert pred.shape == target.shape == ROI.shape
    # print(torch.sigmoid(pred).max(), ROI.max(), target.max())
    # pred = torch.sigmoid(pred)[ROI>0.9] hu xi 
    pred = pred[ROI>0.9]
    target = target.float()[ROI>0.9]
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection 
    iou = (intersection + 1e-6) / (union + 1e-6)
    return 1 - iou

loss_fn = iou_loss

wandb.init(project="Remeow")
wandb.define_metric("pstep")
logger = wandb
model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load("model.pth"))
trainer = trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn)


# %%
trainer.train()


