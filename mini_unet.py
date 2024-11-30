import torch
import torch.nn as nn

# Define the MiniUNet model again to ensure all imports are correctly set up
class MiniUNet(nn.Module):
    def __init__(self, in_channels=33, out_channels=1):
        super(MiniUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        up1 = self.up1(b)
        d1 = self.dec1(torch.cat([up1, e2], dim=1))
        up2 = self.up2(d1)
        d2 = self.dec2(torch.cat([up2, e1], dim=1))
        
        # Final layer
        out = self.final(d2)
        return out


if __name__ == "__main__":
    input_shape = (1, 33, 128, 128)  # Batch size 1, 33 input channels, 128x128 spatial dimensions
    sample_input = torch.randn(input_shape)

    model = MiniUNet(in_channels=33, out_channels=1)

    output = model(sample_input)

    input_shape, output.shape
