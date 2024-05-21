import torch.nn as nn


class H14_NSFW_Detector(nn.Module):
    def __init__(self, clip_dim=1024, out_class: int = 5):
        super().__init__()
        self.input_size = clip_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, out_class)
        )

    def forward(self, x):
        return self.layers(x)
