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
        self.proj = torch.nn.Conv2d(768 * 2, 128, 1)
        # u-net like decoder 
        self.unet = UNet(in_channels=128, out_channels=8)
        
        for param in self.backbone.parameters(): 
            param.requires_grad = False
            
        self.head = torch.nn.Sequential(
            nn.Conv2d(8, 1, 3),
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

        in_feature1 = diff1 * in_feature
        in_feature2 = diff2 * in_feature
        
        pred = self.unet(torch.nn.functional.relu(self.proj(torch.cat([in_feature1, in_feature2], dim=1))))
        pred = self.head(pred)
        pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode="bilinear", align_corners=False)
        return pred

if __name__ == "__main__":
    model = MyModel(None)
    x = torch.randn(2, 9, 512, 512)
    print(model(x).shape)