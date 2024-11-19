from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch
import numpy as np
from PIL import Image



class MyModel(torch.nn.Module):
    def __init__(self, args): 
        super().__init__()
        config = SegformerConfig.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.drop_path_rate = args.drop_path_rate
        config.classifier_dropout = args.classifier_dropout
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", config=config)
        self.backbone.decode_head.classifier = torch.nn.Identity()
        self.decoder = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(256 * 3, 256, kernel_size=(9, 9)),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1, kernel_size=(9, 9)),  
            torch.nn.Sigmoid()         
        )
        
        # self.proj = torch.nn.Sequential(
        #     torch.nn.Dropout2d(0.1), 
        #     torch.nn.Conv2d(256, 1, kernel_size=(1 ,1), padding="same"),
        #     torch.nn.Sigmoid() # That is why same code not training and untill i saw loss higher than 1 i relizewd 
        # )
        
        for param in self.backbone.parameters(): 
            param.requires_grad = False 
             
            
    
    def forward(self, X):
        in_img, long_img, short_img = X[:, :3], X[:, 3:6], X[:, 6:]
        in_pred = self.backbone(in_img).logits 
        long_pred = self.backbone(long_img).logits
        short_pred = self.backbone(short_img).logits
        
        pred = torch.cat([in_pred, long_pred, short_pred], dim=1)
        res = self.decoder(pred)
        pred = torch.nn.functional.interpolate(res, size=(512, 512), mode='nearest')
        return pred
    
if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.attention_probs_dropout_prob = 0.1
            self.hidden_dropout_prob = 0.1
            self.drop_path_rate = 0.1
            self.classifier_dropout = 0.1
            self.ksteps = 200 


    model = MyModel(Args())
    model.eval()
    input_tensor = torch.rand(8, 9, 512, 512)
    output = model(input_tensor)
    print(output.shape)
    
    
    
