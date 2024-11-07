# %%
class Args:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.15
        self.hidden_dropout_prob = 0.15
        self.drop_path_rate = 0.15
        self.classifier_dropout = 0.1
        self.ksteps = 1000 

args = Args()

# %%
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image
from model import MyModel
from trainer import trainer

from dataloader import CustomDataset
import wandb

model = MyModel(args)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1) 
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ksteps)
train_dataset = CustomDataset("/mnt/fastdata/preaug_cdnet/", "/mnt/fastdata/CDNet", 1, "train")
val_dataset = CustomDataset("/mnt/fastdata/preaug_cdnet/", "/mnt/fastdata/CDNet", 1, "val")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True, persistent_workers=True, prefetch_factor=20) 
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True, persistent_workers=True, prefetch_factor=20) 

def iou_loss(pred, target):
    assert pred.shape == target.shape, "Shapes of prediction and target should be the same; got %s and %s" % (pred.shape, target.shape)
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / union
    return 1 - iou

loss_fn = iou_loss

wandb.init(project="Remeow")
wandb.define_metric("pstep")
logger = wandb
model = torch.nn.DataParallel(model).cuda()
trainer = trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, loss_fn)


# %%
trainer.train()


