from transformers import BeitForSemanticSegmentation
import torch.nn as nn
import torch
backbone = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")


class BCA(nn.Module):
    def __init__(self):
        super(BCA, self).__init__()
        self.q_proj = nn.Conv2d(192, 256, 1)
        self.k_proj = nn.Conv2d(192, 256, 1)
        self.v_proj = nn.Conv2d(192, 256, 1)
        self.softmax = nn.Softmax(dim=1)
        self.norm1 = nn.LayerNorm(256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 768, 1),
            nn.ReLU(),
            nn.Linear(768, 256, 1)
        )
        self.norm2 = nn.LayerNorm(256)

    def forward(self, q, k, v):
        B, _, H, W = q.shape
        q = self.q_proj(q) # q.shape = (B, 512, H, W)
        k = self.k_proj(k) # k.shape = (B, 512, H, W)
        v = self.v_proj(v) # v.shape = (B, 512, H, W)

        # todo: try reduce q,k into lower resolution
        q = q.reshape(q.shape[0], q.shape[1], -1).permute(0, 2, 1) # q.shape = (B, H*W, 512)
        k = k.reshape(k.shape[0], k.shape[1], -1).permute(0, 2, 1) # k.shape = (B, H*W, 512)
        v = v.reshape(v.shape[0], v.shape[1], -1).permute(0, 2, 1) # v.shape = (B, H*W, 512)

        attn = torch.einsum("bld, bld->bl", q, k) # attn.shape = (B, H*W)
        attn = self.softmax(attn)

        out = torch.einsum("bl, bld->bld", attn, v) # v.shape = (B, H*W, 512)
        out = self.norm1(out + v) # v.shape = (B, H*W, 512)
        out = self.norm2(out + self.mlp(out))

        out = out.permute(0, 2, 1).reshape(out.shape[0], out.shape[2], H, W) # v.shape = (B, 512, H, W)
        return out



class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.backbone = backbone
        self.backbone.decode_head.classifier = nn.Identity()
        for param in self.backbone.parameters(): 
            param.requires_grad = False

        self.bcas = torch.nn.ModuleList([BCA() for _ in range(4)])

        self.head = nn.Conv2d(512, 1, 1)
    def forward(self, X):
        X = torch.nn.functional.interpolate(X, size=(320, 320), mode="bilinear", align_corners=False)
        in_img = X[:, :3]
        long_img = X[:, 3:6]
        short_img = X[:, 6:9]
        
        X = torch.cat([in_img, long_img, short_img], dim=0)

        for bca in self.bcas:
            # in_feature = self.backbone(in_img).logits
            # long_feature = self.backbone(long_img).logits
            # short_feature = self.backbone(short_img).logits 
            X_features = self.backbone(X).logits #nowonder it is slow
            in_feature, long_feature, short_feature = torch.split(X_features, len(X_features)//3, dim=0)
            
            pred = bca(in_feature, long_feature, short_feature)
            in_feature = pred



        pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode="bilinear", align_corners=False)
        return self.head(pred)

if __name__ == "__main__":
    model = MyModel(None)
    x = torch.randn(2, 9, 512, 512)
    print(model(x).shape)