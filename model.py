import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ConvNextClassifier, self).__init__()
        
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_tiny(weights=weights)
        else:
            self.backbone = models.convnext_tiny(weights=None)
        
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Flatten(1),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes=3, pretrained=True):
    model = ConvNextClassifier(num_classes=num_classes, pretrained=pretrained)
    return model

if __name__ == "__main__":
    model = create_model()
    x = torch.randn(2, 3, 1024, 1024)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")