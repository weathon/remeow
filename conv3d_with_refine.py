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
    backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(17 if conv3d else 12, in_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
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
            nn.ReLU(),
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
        
        # self.refine_gru = ConvGRU(input_size=(512, 512),
        #         input_dim=7,
        #         hidden_dim=32,
        #         kernel_size=(3, 3),
        #         num_layers=5,
        #         dtype=torch.cuda.FloatTensor,
        #         batch_first=True,
        #         bias = True, 
        #         return_all_layers = False)
        # self.out_linear = torch.nn.Conv3d(64, 1, (1, 1, 1))

        # add noise before refine? 
        # self.refine_conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(33, 64, 3),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout2d(0.15), 
        #     torch.nn.Conv2d(64, 32, 3),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout2d(0.15),
        #     torch.nn.Conv2d(32, 8, 3),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout2d(0.15),
        #     torch.nn.Conv2d(8, 1, 3),
        # )  
        self.refine_conv = MiniUNet(in_channels=64 + 32, out_channels=64)
        frame_dim = 16 if self.args.refine_see_bg else 32
        self.frame_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding="same"),
            torch.nn.AvgPool2d(2), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, frame_dim, 3, padding="same"),
            torch.nn.AvgPool2d(2),
            torch.nn.ReLU(),
        )

    def refine(self, mask, current, long):
        """ 
        Input: mask, current
        Output: Refined Mask
        """
        masks = [self.head(self.upsample(mask))]
        current = self.frame_encoder(current)
        if self.args.refine_see_bg:
            long = self.frame_encoder(long)
            
        for i in range(self.args.refine_steps - 1): 
            noise = torch.randn_like(mask) * torch.std(mask) * self.args.noise_level
            mask = noise.detach() + mask # += not working but = + is okay 
            if self.args.refine_see_bg:
                X = torch.cat([mask, current, long], dim=1)
            else:
                X = torch.cat([mask, current], dim=1)
            delta_mask = self.refine_conv(X) 
            delta_mask = torch.nn.functional.interpolate(delta_mask, size=(128, 128), mode="nearest")
            if self.args.refine_mode == "residual":
                mask = mask + delta_mask
            else:
                mask = delta_mask
            masks.append(self.head(self.upsample(mask))) 
        
        return torch.cat(masks, dim=1)
    def forward(self, X): 
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
        X = self.backbone(X).logits 
        assert X.shape[1:] == (64, 128, 128), X.shape



        masks = self.refine(X, current, long)
        # print(masks.shape)
        
        # masks = self.upsample(masks) 
        masks = torch.sigmoid(masks)
        return masks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--refine_mode', type=str, default="residual", help='Refine mode', choices=["residual", "direct"])
    parser.add_argument('--noise_level', type=float, default=1, help='Noise level') 
    parser.add_argument('--mask_upsample', type=str, default="interpolate", help='Mask upsample method', choices=["interpolate", "transpose_conv", "shuffle"])
    parser.add_argument('--refine_see_bg', action="store_true", help='If refine operator can see background')

    args = parser.parse_args()
    model = MyModel(args)
    from video_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/home/wg25r/fastdata/CDNet", "/home/wg25r/fastdata/CDNet", 3, "train")[0]
    pred = model(X[None])
    print(pred.shape)
    print(pred.min(), pred.max())
    pred.mean().backward()
        