from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation

from convGRU import ConvGRU
from mini_unet import MiniUNet
conv3d = True

def get_backbone(n):
    n = int(n)
    in_dim = [32, 64, 64, 64, 64][n]
    out_dim = [256, 256, 768, 768, 768][n]
    backbone = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b{n}-finetuned-ade-512-512")
    backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(17 + 32 if conv3d else 12 + 32, in_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
    backbone.decode_head.classifier = torch.nn.Conv2d(out_dim, 64, kernel_size=(1, 1), stride=(1, 1))
    return backbone

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__() 
        self.args = args
        self.backbone = get_backbone(args.backbone)
        self.head = torch.nn.Sequential(
            nn.Conv2d(32, 16, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding="same"), 
        ) 
        self.conv3d = torch.nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3), padding="same"),
            nn.Dropout3d(0.1),
            nn.ReLU(),
            nn.Conv3d(8, 16, (3, 3, 3), padding="same"),
            nn.Dropout3d(0.1), 
            nn.ReLU(),
            nn.Conv3d(16, 8, (3, 3, 3), padding="same"), 
            nn.Dropout3d(0.1),
        ) if conv3d else torch.nn.Identity()

        self.t_dim = nn.Linear(8, 1)
        if self.args.mask_upsample == "interpolate":
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(size=(512, 512), mode="bilinear"),
                torch.nn.Conv2d(64, 32, 3, padding="same"),
                torch.nn.ReLU()
            )
        elif self.args.mask_upsample == "transpose_conv":
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, 2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding="same"),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, 2),
                nn.ReLU(),
                torch.nn.Upsample(size=(512, 512))
            )    
        else:
            self.upsample = torch.nn.Sequential(
                nn.PixelShuffle(4),
                nn.Conv2d(4, 32, 3, padding="same"),
                nn.ReLU()
            )
        
        
        # self.hist_encoder = torch.nn.Sequential(
        #     torch.nn.Conv3d(3, 16, (3, 3, 3), padding="same"),
        #     torch.nn.Dropout3d(0.1),
        #     torch.nn.MaxPool3d((1, 2, 2)),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv3d(16, 32, (3, 3, 3), padding="same"),
        #     torch.nn.Dropout3d(0.1),
        #     torch.nn.MaxPool3d((1, 2, 2)),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv3d(32, 32, (51, 1, 1)),
        #     torch.nn.Dropout3d(0.1),
        #     torch.nn.MaxPool3d((1, 2, 2)),
        # ) 
        self.hist_encoder = torch.nn.Sequential(
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(153 + 3, 128, 1, padding="same"),
            torch.nn.Dropout2d(0.3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, 1, padding="same"),
            torch.nn.Dropout2d(0.3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 1, padding="same"),
            torch.nn.Dropout2d(0.3),
        )
        
    def forward(self, X): 
        # print("inside ", X.shape) 
        X, hist = X[:,:30], X[:,30:]
        # hist = hist.reshape(-1, 51, 3, 512, 512).permute(0, 2, 1, 3, 4)
        hist = hist/hist.sum()
        if self.training:
            hist = hist + torch.randn_like(hist) * 0.1 # or should i do 1..t model 
        # print(hist_features.shape)
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]        
        current = frames[:, :3]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        frames = self.conv3d(frames)
        
        hist = torch.cat([hist, current], dim=1)
        hist_features = self.hist_encoder(hist)
        # print("a" * 100)
        hist_features = torch.nn.functional.interpolate(hist_features, size=(512, 512), mode="nearest")
        # print(hist_features.shape) 
        # print("b" * 100)
        assert frames.shape[1:] == (8 if conv3d else 3, 8, 512, 512), frames.shape
        frames = frames.permute(0, 1, 3, 4, 2)
        assert frames.shape[1:] == (8 if conv3d else 3, 512, 512, 8)
        frames = self.t_dim(frames).squeeze(-1)
        assert frames.shape[1:] == (8 if conv3d else 3, 512, 512)
        X = torch.cat([frames, short, long, current], dim=1)
        assert X.shape[1:] == (17 if conv3d else 12, 512, 512)
        X = torch.cat([X, hist_features], dim=1)
        assert X.shape[1:] == (17 + 32 if conv3d else 12 + 32, 512, 512)
        # print("c" * 100)
        X = self.backbone(X).logits 
        # print("d" * 100)
        assert X.shape[1:] == (64, 128, 128), X.shape

        mask = self.upsample(X) 
        # print("e" * 100)
        mask = self.head(mask)
        # print("f" * 100)
        mask = torch.sigmoid(mask)
        return mask
    
def iou_loss(pred, target, ROI): 
    pshape = pred.shape[:1] + pred.shape[2:]
    assert pred.shape[1] == args.refine_steps, f"pred shape: {pred.shape}"

    assert pshape == target.shape == ROI.shape, f"pred shape: {pshape}, target shape: {target.shape}, ROI shape: {ROI.shape}"
    # print(torch.sigmoid(pred).max(), ROI.max(), target.max())
    # pred = torch.sigmoid(pred)[ROI>0.9] hu xi 
    
    total_loss = 0
    for i in range(args.refine_steps):
        pred_ = pred[:,i][ROI>0.9]
        target_ = target.float()[ROI>0.9] 
        intersection = (pred_ * target_).sum()
        union = pred_.sum() + target_.sum() - intersection 
        iou = (intersection + 1e-6) / (union + 1e-6)
        total_loss +=  (1 - iou) * (0.6 ** (args.refine_steps - i - 1))
        
    return total_loss

if __name__ == "__main__": 
    import argparse
    import shlex
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
    parser.add_argument('--background_type', type=str, default="mog2", help='Background type', choices=["mog2", "sub"])
    argString = '--fold 2 --steps 10000 --learning_rate 8e-5 --weight_decay 4e-3 --background_type sub --refine_step 1 --backbone 0'
    args = parser.parse_args(shlex.split(argString))
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 
    model = MyModel(args)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    from video_histgram_dataloader import CustomDataset
    import torch
    # X, Y, ROI = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "train")[0]
    # X = torch.randn(9, 183, 512, 512).cuda()
    dataloader = torch.utils.data.DataLoader(CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "train"), batch_size=9, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    # model.train()
    # optimizer.zero_grad()
    # X, Y, ROI = next(iter(dataloader))
    # X = X.cuda().to("cuda:0") 
    # pred = model(X)
    # print(pred.shape)
    # print(pred.min(), pred.max())
    # pred.mean().backward()
    # optimizer.step()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps) 
    
    train_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "train")
    val_dataset = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "val")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=9, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=9, shuffle=True, num_workers=30, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    
    import wandb
    wandb.init(project="Remeow", config=args)
    wandb.define_metric("pstep")
    logger = wandb
    from trainer import Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, iou_loss, args)
    trainer.train()
