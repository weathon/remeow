from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation

from convGRU import ConvGRU
conv3d = True
backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(17 if conv3d else 12, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
backbone.decode_head.classifier = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__() 
        self.args = args
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 128, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding="same"),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding="same"),
            nn.Dropout2d(0.15),
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
        self.refine_gru = ConvGRU(input_size=(640, 640),
                input_dim=7,
                hidden_dim=64,
                kernel_size=(3, 3),
                num_layers=5,
                dtype=torch.FloatTensor,
                batch_first=True,
                bias = True, 
                return_all_layers = False)
        self.out_linear = torch.nn.Conv3d(64, 1, (1, 1, 1))
        # self.refine_conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(7, 8, 5),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Conv2d(8, 16, 5),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Conv2d(16, 8, 5),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Conv2d(8, 1, 5),
        # )
        
    def refine(self, mask, current, long):
        """
        Input: mask, current
        Output: Refined Mask
        """
        
        X = torch.cat([mask, current, long], dim=1)
        X = X.unsqueeze(1).repeat(1, 5, 1, 1, 1)
        delta_X = self.refine_gru(X)
        states = delta_X[0][0]
        states = states.permute(0, 2, 1, 3, 4)
        delta_masks = self.out_linear(states).squeeze(1)
        # print(mask.shape)
        mask = mask.repeat(1, 5, 1, 1)
        assert delta_masks.shape == mask.shape
        masks = delta_masks + mask
      
        return masks
    def forward(self, X): 
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        frames = self.conv3d(frames)
        assert frames.shape[1:] == (8 if conv3d else 3, 8, 640, 640), frames.shape
        frames = frames.permute(0, 1, 3, 4, 2)
        assert frames.shape[1:] == (8 if conv3d else 3, 640, 640, 8)
        frames = self.t_dim(frames).squeeze(-1)
        assert frames.shape[1:] == (8 if conv3d else 3, 640, 640)
        X = torch.cat([frames, short, long, current], dim=1)
        assert X.shape[1:] == (17 if conv3d else 12, 640, 640)
        X = self.backbone(X).logits 
        assert X.shape[1:] == (512, 160, 160)
        X = self.upsample(X)
        assert X.shape[1:] == (512, 640, 640)
        pred = X
        mask = self.head(pred)
        
        masks = []
        mask = self.refine(mask, current, long)
        masks.append(torch.sigmoid(mask))
        
        return torch.cat(masks, dim=1)

if __name__ == "__main__":
    model = MyModel(None)
    from video_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "train")[0]
    print(model(X[None]).shape)
    
        