from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation
backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
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
        self.conv3d = torch.nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(8, 16, (3, 3, 3), padding="same"), 
            nn.ReLU(),
            nn.Conv3d(16, 9, (3, 3, 3), padding="same"), 
            nn.ReLU(),
        )
        self.t_dim = nn.Linear(10, 1)
        self.upsample = torch.nn.Upsample(scale_factor=4, mode="bicubic")
        
    def forward(self, X): 
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        frames = self.conv3d(frames)
        long = long.unsqueeze(2)
        short = short.unsqueeze(2)
        X = torch.cat([frames, short, long], dim=2)
        # assert X.shape[1:] == (3, 10, 640, 640)
        X = self.conv3d(X)
        # assert X.shape[1:] == (16, 10, 640, 640)
        X = X.permute(0, 1, 3, 4, 2)
        # assert X.shape[1:] == (16, 640, 640, 10)
        X = self.t_dim(X).squeeze(-1)
        # assert X.shape[1:] == (16, 640, 640)
        X = self.backbone(X).logits
        # assert X.shape[1:] == (512, 160, 160)
        X = self.upsample(X)
        # assert X.shape[1:] == (512, 640, 640)
        pred = X
        return self.head(pred)

if __name__ == "__main__":
    model = MyModel(None)
    from video_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "train")[0]
    print(model(X[None]).shape)
    
        