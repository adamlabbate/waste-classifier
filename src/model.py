import torch
import torch.nn as nn


class WasteClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dropout = nn.Dropout(p=0.5)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 6)

    def forward(self, x):

      x = self.block1(x)
      x = self.block2(x)
      x = self.block3(x)

      x = self.global_avg_pool(x)
      x = x.view(x.size(0), -1)
      x = self.dropout(x)
      x = self.fc(x)
      return x
    
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']