from transformers import BeitForSemanticSegmentation, SegformerForSemanticSegmentation
import torch.nn as nn
import torch
from unet import UNet




class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.backbone.decode_head.classifier = torch.nn.Identity()

        # u-net like decoder
        # self.unet = UNet(in_channels=768*2, out_channels=256)
        # for param in self.backbone.parameters(): 
        #     param.requires_grad = False
            
        self.head = torch.nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3),
            nn.Sigmoid()
        )
        self.c = torch.nn.Parameter(torch.tensor(1.0))
        
    def forward(self, X):
        X = torch.nn.functional.interpolate(X, size=(512, 512), mode="bilinear", align_corners=False)
        in_img = X[:, :3]
        long_img = X[:, 3:6]
        short_img = X[:, 6:9]
        
        in_feature = self.backbone(in_img).logits
        long_feature = self.backbone(long_img).logits
        short_feature = self.backbone(short_img).logits
        
        diff1 = 1 - torch.nn.functional.cosine_similarity(in_feature, long_feature, dim=1, eps=1e-8).unsqueeze(1)
        diff2 = 1 - torch.nn.functional.cosine_similarity(in_feature, short_feature, dim=1, eps=1e-8).unsqueeze(1)

        pred = self.head(torch.cat([diff1, diff2], dim=1))

        pred = torch.nn.functional.interpolate(diff1, size=(512, 512), mode="bilinear", align_corners=False)
        return torch.sigmoid(pred) 

if __name__ == "__main__":
    model = MyModel(None)
    x = torch.randn(2, 9, 512, 512)
    print(model(x).shape)