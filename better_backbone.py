from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation
backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
backbone.decode_head.classifier = torch.nn.Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__() 
        self.args = args
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding="same"), 
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding="same"),
            nn.Sigmoid() 
        )
             
        
    def forward(self, X):
        X = X.contiguous()
        X = torch.nn.functional.interpolate(X, size=(640, 640), mode="bilinear", align_corners=False)
        X_features = self.backbone(X).logits 
        X_features = X_features.contiguous()
        pred = torch.nn.functional.interpolate(X_features, size=(640, 640), mode="bilinear", align_corners=False)
        return self.head(pred)

if __name__ == "__main__":
    model = MyModel(None)
    x = torch.randn(2, 9, 512, 512)
    print(model(x).shape)