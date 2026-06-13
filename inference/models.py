import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet v2 - arhitectura compacta pentru EEG, adaptata pentru seizure prediction.
    F1=32 filtre temporale, D=6 multiplicator depthwise, F2=64 filtre separabile.
    """
    def __init__(self, n_channels=18, n_samples=1024,
                 F1=32, D=6, F2=64, dropout=0.4):
        super().__init__()
        # Block 1: convolutie temporala
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1))
        # Block 2: convolutie spatiala depthwise
        self.b2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout))
        # Block 3: convolutie separabila
        self.b3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout))
        with torch.no_grad():
            n_out = self.b3(self.b2(self.b1(
                torch.zeros(1, 1, n_channels, n_samples)))).numel()
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_out, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.clf(self.b3(self.b2(self.b1(x)))).squeeze(1)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
