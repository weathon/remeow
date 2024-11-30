# https://github.com/haofeixu/gmflow/blob/main/gmflow/matching.py

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image
import torchvision


# %% 

class MyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.backbone.decode_head.classifier = torch.nn.Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1))
        
        self.downsample = torch.nn.AvgPool2d(2)
        self.fusion = torch.nn.MultiheadAttention(128, 8, dropout=0.1)
        # torch.nn.TransformerDecoderLayer(d_model=128, nhead=8, batch_first=True)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Upsample(size=(640, 640)),
            torch.nn.Conv2d(16, 1, 3, padding="same"),
            torch.nn.Sigmoid()
        )
    def forward(self, X, return_flow=False):
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        current_feature = self.backbone(current).logits
        current_feature = self.downsample(current_feature)
        current_feature = current_feature.flatten(-2, -1).permute(0, 2, 1)
        
        short_feature = self.backbone(short).logits
        short_feature = self.downsample(short_feature)
        short_feature = short_feature.flatten(-2, -1).permute(0, 2, 1)
        
        long_feature = self.backbone(long).logits
        long_feature = self.downsample(long_feature)
        long_feature = long_feature.flatten(-2, -1).permute(0, 2, 1)
        
        fused1 = self.fusion(short_feature, current_feature, current_feature)[0]
        fused2 = self.fusion(long_feature, current_feature, current_feature)[0]
        
        fused = torch.cat([fused1, fused2], dim=-1).reshape(-1, 80, 80, 256).permute(0, 3, 1, 2)
        fused = self.decoder(fused)
        return fused
if __name__ == "__main__": 
    # %%
    from video_dataloader import CustomDataset
    import cv2
    class Args:
        def __init__(self):
            self.attention_probs_dropout_prob = 0.05
            self.hidden_dropout_prob = 0.05
            self.drop_path_rate = 0.05
            self.classifier_dropout = 0
            self.ksteps = 200 


    args = Args()
    model = MyModel(args)
    X = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "val")[0][0].unsqueeze(0)
    pred = model(X, return_flow=True)
    print(pred.shape)

    
    




    
