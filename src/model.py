from torch import nn
from torchvision import models


class CNNBaseline(nn.Module):
    def __init__(self , num_classes=8 , dropout_p=0.0 , pool_type='max'):
        super().__init__()
        assert pool_type in ('max' , 'avg')
        pool = nn.MaxPool2d if pool_type == 'max' else nn.AvgPool2d

        def block(c_in , c_out):
            return nn.Sequential(
            nn.Conv2d(c_in , c_out , kernel_size=3 , padding= 1),
            nn.ReLU(),
            pool (2)
        )
        
        self.features = nn.Sequential(block(3 , 32), block(32 , 64) ,  block(64 , 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(128 , num_classes)
        )

    def forward(self , x):
        return self.classifier(self.gap(self.features(x)))
    
class TLResNet(nn.Module):
    def __init__(self, num_classes=8, trainable=('fc',)):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(512, num_classes)
        self.set_trainable(trainable)

    def set_trainable(self, names):
        self.trainable = tuple(names)
        for n, p in self.net.named_parameters():
            p.requires_grad = any(n.startswith(t) for t in self.trainable)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for n, m in self.net.named_children():
                if n not in self.trainable:
                    m.eval()          
        return self

    def forward(self, x):
        return self.net(x)


def describe_run(model, opt):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable {train:,} / {total:,}')
    for i, g in enumerate(opt.param_groups):
        print(f'group {i}: lr={g["lr"]}, params={sum(p.numel() for p in g["params"]):,}')

class ResNetFromScratch(nn.Module):
    """ResNet-18 from scratch, adapted for small images (32×32)."""
    
    def __init__(self, num_classes=8):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.net(x)