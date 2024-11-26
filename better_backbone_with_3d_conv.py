from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation
backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
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
        self.flow_extractor = torch.nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=(1, 0, 0)),
            nn.ReLU(),
            nn.AvgPool3d((1, 2, 2)),
            nn.Conv3d(32, 16, 3, padding=(1, 0, 0)),
            nn.AvgPool3d((1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(16, 1, 3, padding=(1, 0, 0)),  
        )
        self.t_dim = nn.Conv2d(4, 1, 1)
        
    def forward(self, X): 
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:,:3] 
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        lowres_frames = torch.nn.functional.interpolate(frames, size=(4, 224, 224), mode="trilinear")
        flow = self.flow_extractor(lowres_frames).squeeze(1)
        flow = self.t_dim(flow)
        flow = torch.nn.functional.interpolate(flow, size=(640, 640), mode="bilinear")
        X = torch.cat([long, short, current, flow], dim=1)
        X_features = self.backbone(X).logits 
        pred = torch.nn.functional.interpolate(X_features, size=(640, 640), mode="bicubic")
        return self.head(pred)

if __name__ == "__main__":
    model = MyModel(None)
    from video_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "train")[0]
    print(model(X[None]).shape)
    
        