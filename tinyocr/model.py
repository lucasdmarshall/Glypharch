"""CNN recognizer with CTC. No recurrent layer: the line is one convolution.

A BiLSTM on top of the CNN is the usual CRNN piece, and it is also the
piece that adds latency. For a fixed 32x160 line the conv stack already
emits one feature vector per time step, so greedy CTC can read it directly.
"""

import torch
from torch import nn

from tinyocr.config import DETAIL_H, IMG_H


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TinyRecognizer(nn.Module):
    def __init__(self, nclass: int):
        super().__init__()
        if nclass < 2:
            raise ValueError("need a blank plus at least one character")
        self.cnn = nn.Sequential(
            ConvBNAct(1, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.fc = nn.Linear(128, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input height is fixed at 32, so the last map is exactly one row.
        y = self.cnn(x)
        y = y[:, :, 0, :]
        y = y.transpose(1, 2)
        return self.fc(y).log_softmax(dim=-1)


class DetailRecognizer(nn.Module):
    """TinyRecognizer plus one height pool, for a 64px line.

    The extra pool has no parameters. A 32px checkpoint loads into the
    conv and linear weights. Width is still pooled twice, so a 160px
    line is still 40 time steps.
    """

    def __init__(self, nclass: int):
        super().__init__()
        if nclass < 2:
            raise ValueError("need a blank plus at least one character")
        self.cnn = nn.Sequential(
            ConvBNAct(1, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.fc = nn.Linear(128, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn(x)
        y = y[:, :, 0, :]
        y = y.transpose(1, 2)
        return self.fc(y).log_softmax(dim=-1)


def recognizer_for(height: int, nclass: int) -> nn.Module:
    if height == IMG_H:
        return TinyRecognizer(nclass)
    if height == DETAIL_H:
        return DetailRecognizer(nclass)
    raise ValueError(f"unsupported input height {height}")
