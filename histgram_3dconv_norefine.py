from transformers import BeitForSemanticSegmentation, SegformerConfig
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation
import sys
sys.path.append("./DIS/IS-Net")
from models import *


def get_backbone(n, dropout=0.1, hist_dim = 32, recent_frames="conv3d"):
        n = int(n)
        # source_dim = 17 + hist_dim if recent_frames=="conv3d" else 12 + hist_dim
        if recent_frames == "conv3d":
            source_dim = 17 + hist_dim
        elif recent_frames == "linear":
            source_dim = 12 + hist_dim
        else:
            source_dim = 9 + hist_dim
        if n != 5:
            in_dim = [32, 64, 64, 64, 64][n]
            out_dim = [256, 256, 768, 768, 768][n]
            config = SegformerConfig.from_pretrained(f"nvidia/segformer-b{n}-finetuned-ade-512-512")
            config.attention_probs_dropout_prob = dropout
            config.hidden_dropout_prob = dropout
            backbone = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b{n}-finetuned-ade-512-512", config=config)
            backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(source_dim, in_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
            backbone.decode_head.classifier = torch.nn.Conv2d(out_dim, 64, kernel_size=(1, 1), stride=(1, 1))
            return backbone
        else:
            in_dim = 64
            out_dim = 768
            config = SegformerConfig.from_pretrained(f"nvidia/segformer-b5-finetuned-ade-640-640")
            config.attention_probs_dropout_prob = dropout
            config.hidden_dropout_prob = dropout
            backbone = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b5-finetuned-ade-640-640", config=config)
            backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(source_dim, in_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
            backbone.decode_head.classifier = torch.nn.Conv2d(out_dim, 64, kernel_size=(1, 1), stride=(1, 1))
            return backbone
    
    
from peft import LoraConfig, TaskType
from peft import get_peft_model
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
class MyModel(nn.Module):
    def __init__(self, args, softmax=True):
        super(MyModel, self).__init__() 
        self.args = args 
        self.softmax = softmax
        self.backbone = get_backbone(args.backbone, 0.1, hist_dim=32 if args.histogram else 0, recent_frames=self.args.recent_frames)
        if args.lora:
            config = LoraConfig(
                r=128,
                lora_alpha=128,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense1", "dense2", "dwconv.dwconv"],
                bias="none",
            ) 
            self.backbone = get_peft_model(self.backbone, config)
            print_trainable_parameters(self.backbone) 
        self.head = torch.nn.Sequential(
            nn.Conv2d(32, 16, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(8, self.args.num_classes, 3, padding="same"),
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
        ) if self.args.recent_frames == "conv3d" else torch.nn.Identity()

        self.t_dim = nn.Linear(8, 1)
        if self.args.mask_upsample == "interpolate":
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(size=(self.args.image_size, self.args.image_size), mode="bilinear"),
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
                torch.nn.Upsample(size=(self.args.image_size, self.args.image_size))
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
        # hist = hist.reshape(-1, 51, 3, self.args.image_size, self.args.image_size).permute(0, 2, 1, 3, 4)
        if self.args.histogram:
            hist = hist/(hist.sum(dim=1, keepdim=True) + 1e-6)
            if self.training:
                hist = hist + torch.randn_like(hist) * hist.std() * 0.05 
        # print(hist_features.shape)
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]        
        current = frames[:, :3]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        frames = self.conv3d(frames)
        if self.args.histogram:
            hist = torch.cat([hist, current], dim=1)
            hist_features = self.hist_encoder(hist)
            # print("a" * 100)
            hist_features = torch.nn.functional.interpolate(hist_features, size=(self.args.image_size, self.args.image_size), mode="nearest")
        # print(hist_features.shape) 
        # print("b" * 100)
        if self.args.recent_frames != "none":
            assert frames.shape[1:] == (8 if self.args.recent_frames == "conv3d" else 3, 8, self.args.image_size, self.args.image_size), frames.shape
            frames = frames.permute(0, 1, 3, 4, 2)
            assert frames.shape[1:] == (8 if self.args.recent_frames == "conv3d" else 3, self.args.image_size, self.args.image_size, 8)
            frames = self.t_dim(frames).squeeze(-1)
            assert frames.shape[1:] == (8 if self.args.recent_frames == "conv3d" else 3, self.args.image_size, self.args.image_size)
            X = torch.cat([frames, short, long, current], dim=1)
            assert X.shape[1:] == (17 if self.args.recent_frames == "conv3d" else 12, self.args.image_size, self.args.image_size)
        else:
            X = torch.cat([short, long, current], dim=1)
            assert X.shape[1:] == (9, self.args.image_size, self.args.image_size)
            
            
        if self.args.histogram:
            X = torch.cat([X, hist_features], dim=1)
        
        # assert X.shape[1:] == (17 + 32 if conv3d else 12 + 32, self.args.image_size, self.args.image_size)
        # print("c" * 100)
        X = self.backbone(X).logits 
        # print("d" * 100)
        # assert X.shape[1:] == (64, 128, 128), X.shape

        mask = self.upsample(X) 
        # print("e" * 100)
        mask = self.head(mask)
        # print("f" * 100)
        assert mask.shape[1] == self.args.num_classes
        if self.softmax:
            mask = torch.nn.functional.softmax(mask, dim=1)
        else:
            mask = mask

        return mask


class ISNetBackbone(nn.Module):
    def __init__(self, args):
        super(ISNetBackbone, self).__init__()
        self.backbone = ISNetDIS().cuda()
        self.backbone.load_state_dict(torch.load("/home/wg25r/isnet-general-use.pth"))
        self.backbone.conv_in = torch.nn.Identity()
        self.backbone.side1 = torch.nn.Identity()
        self.in_conv = nn.Conv2d(9, 64, kernel_size=(3,3), padding=(1,1))
        self.head = torch.nn.Conv2d(64, 3, 3, 1, 1) # should not include this in the backbone because of the L2 loss

    def forward(self, X, softmax=True):
        X = X[:,:30]
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]        
        current = frames[:, :3]
        x = torch.cat([short, long, current], dim=1)
        x = self.in_conv(x)
        x = self.backbone(x)[1][0]
        x = self.head(x)
        if softmax:
            x = torch.nn.functional.softmax(x, dim=1)
        return x


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
    parser.add_argument('--lora', action="store_true", help='If use LoRA')
    parser.add_argument('--save_name', type=str, default="model", help='Model save name')
    parser.add_argument('--final_weight_decay', type=float, default=3e-2, help='Final weight decay')
    parser.add_argument('--use_difference', action="store_true", help='If use difference between current and background ratehr than background frame')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--recent_frames', type=str, default="conv3d", help='Recent frames method', choices=["conv3d", "linear", "none"])

    argString = '--fold 2 --steps 25000 --learning_rate 3e-5 --weight_decay 2e-2 --background_type mog2 --refine_step 1 --backbone 4 --image_size 512 --gpu 1 --clip 2 --conf_penalty 0.05 --lambda2 100 --hard_shadow --save_name 2 --final_weight_decay 4e-2 --final_weight_decay 5e-2 --save_name 6 --print_every 1000 --recent_frames conv3d'
    
    args = parser.parse_args(shlex.split(argString))
    import os
    def regularization_loss(model_0, model_t):
        total_loss = 0
        count = 0
        for param_0, param_t in zip(model_0, model_t):
            total_loss += (param_0 - param_t).abs().sum()
            count += param_0.size()[0]
            
        return total_loss / count
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 
    model = MyModel(args)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    from video_histgram_dataloader import CustomDataset
    import torch
    # X, Y, ROI = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "train")[0]
    # X = torch.randn(9, 183, self.args.image_size, self.args.image_size).cuda()
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
    trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, logger, iou_loss, args, regularization_loss)
    trainer.train()
