import torch
import torch.nn as nn
from torchvision import models


class TransferLearningEncoder(nn.Module):
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        super(TransferLearningEncoder, self).__init__()
        
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.out_channels = 2048
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.out_channels = 512
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.out_channels = 512
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.features = base_model.features
            self.out_channels = 1280
        elif backbone == 'efficientnet_b3':
            base_model = models.efficientnet_b3(pretrained=pretrained)
            self.features = base_model.features
            self.out_channels = 1536
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone_name = backbone
        print(f"Loaded {backbone} backbone (pretrained={pretrained})")
    
    def forward(self, x):
        return self.features(x)


class TransferECGModel(nn.Module):
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, 
                 num_leads: int = 12, max_seq_len: int = 5000):
        super(TransferECGModel, self).__init__()
        
        self.encoder = TransferLearningEncoder(backbone, pretrained)
        
        from model import SignalDecoder
        self.decoder = SignalDecoder(
            feature_dim=self.encoder.out_channels,
            num_leads=num_leads,
            max_seq_len=max_seq_len
        )
    
    def forward(self, images, target_lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, target_lengths)
        return outputs


def create_transfer_model(backbone: str = 'resnet50', pretrained: bool = True, 
                         device='cpu'):
    model = TransferECGModel(backbone=backbone, pretrained=pretrained)
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    return model


def fine_tune_model(model: nn.Module, unfreeze_layers: int = -1):
    if unfreeze_layers == 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen - only training decoder")
    elif unfreeze_layers > 0:
        modules = list(model.encoder.features.children())
        for i, module in enumerate(modules):
            if i < len(modules) - unfreeze_layers:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        print(f"Unfrozen last {unfreeze_layers} encoder layers")
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("Full model training - all layers unfrozen")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model
