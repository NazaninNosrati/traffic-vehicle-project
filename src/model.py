from torch import nn


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

