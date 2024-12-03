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
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding="same"),
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
                torch.nn.Upsample(scale_factor=4, mode="bicubic"),
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
        
        
        self.hist_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(3, 16, (3, 3, 3), padding="same"),
            torch.nn.Dropout3d(0.1),
            torch.nn.MaxPool3d((1, 2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(16, 32, (3, 3, 3), padding="same"),
            torch.nn.Dropout3d(0.1),
            torch.nn.MaxPool3d((1, 2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(32, 32, (51, 1, 1)),
            torch.nn.Dropout3d(0.1),
            torch.nn.MaxPool3d((1, 2, 2)),
        )
        
    def forward(self, X): 
        X, hist = X[:,:30], X[:,30:]
        hist = hist.reshape(-1, 51, 3, 512, 512).permute(0, 2, 1, 3, 4)
        hist = hist/hist.max()
        hist_features = self.hist_encoder(hist).squeeze(2)
        hist_features = torch.nn.functional.interpolate(hist_features, size=(512, 512), mode="bilinear")
        # print(hist_features.shape)
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        frames = self.conv3d(frames)
        assert frames.shape[1:] == (8 if conv3d else 3, 8, 512, 512), frames.shape
        frames = frames.permute(0, 1, 3, 4, 2)
        assert frames.shape[1:] == (8 if conv3d else 3, 512, 512, 8)
        frames = self.t_dim(frames).squeeze(-1)
        assert frames.shape[1:] == (8 if conv3d else 3, 512, 512)
        X = torch.cat([frames, short, long, current], dim=1)
        assert X.shape[1:] == (17 if conv3d else 12, 512, 512)
        X = torch.cat([X, hist_features], dim=1)
        assert X.shape[1:] == (17 + 32 if conv3d else 12 + 32, 512, 512)
        
        X = self.backbone(X).logits 
        assert X.shape[1:] == (64, 128, 128), X.shape

        mask = self.upsample(X)
        mask = self.head(mask)
        
        return mask

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
    argString = '--gpu 0 --fold 2 --noise_level 0.3 --steps 50000 --learning_rate 4e-5 --mask_upsample shuffle --weight_decay 3e-2'
    args = parser.parse_args(shlex.split(argString))
    model = MyModel(args)
    from video_histgram_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", args, "train")[0]
    pred = model(X[None])
    print(pred.shape)
    print(pred.min(), pred.max())
    pred.mean().backward()
        