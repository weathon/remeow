from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation

from convGRU import ConvGRU
from mini_unet import MiniUNet
conv3d = True
backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(17 if conv3d else 12, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
backbone.decode_head.classifier = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))




class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__() 
        self.args = args
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            nn.Conv2d(64, 32, 3, padding="same"),
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
        self.upsample = torch.nn.Upsample(scale_factor=4, mode="bicubic")
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
        self.frame_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding="same"),
            torch.nn.AvgPool2d(2), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, padding="same"),
            torch.nn.AvgPool2d(2),
            torch.nn.ReLU(),
        )
        
    def refine(self, mask, current, long):
        """ 
        Input: mask, current
        Output: Refined Mask
        """
        masks = [self.head(mask)]
        current = self.frame_encoder(current)
        long = self.frame_encoder(long)
        for i in range(4): 
            noise = torch.randn_like(mask) * torch.std(mask) * 0.1 
            mask = noise.detach() + mask # += not working but = + is okay 
            X = torch.cat([mask, current, long], dim=1)
            delta_mask = self.refine_conv(X) 
            delta_mask = torch.nn.functional.interpolate(delta_mask, size=(128, 128), mode="nearest")
            mask = mask + delta_mask
            masks.append(self.head(mask)) 
        
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
        
        masks = self.upsample(masks) 
        masks = torch.sigmoid(masks)
        return masks

if __name__ == "__main__":
    model = MyModel(None)
    from video_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "train")[0]
    pred = model(X[None])
    print(pred.shape)
    print(pred.min(), pred.max())
    pred.mean().backward()
        