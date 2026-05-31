"""
models.py
=========
Architecture definitions for EEG working memory classification.

Classes
-------
EEGNet              : EEGNet architecture (Lawhern et al., 2018), adapted for
                    multiclass classification with AdaptiveAvgPool2d classifier.
lightweightEEGNet   : Lightweight depthwise-separable CNN architecture, inspired
                    by EEGNet but with a smaller parameter count.
"""

import torch
import torch.nn as nn


# =========================================================
# EEGNET
# =========================================================

class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018) adapted for multiclass EEG decoding.

    Architecture
    ------------
    Block 1 : Temporal Conv2d → BatchNorm → Depthwise spatial Conv2d
              → BatchNorm → ELU → AvgPool → Dropout2d
    Block 2 : Depthwise separable Conv2d → BatchNorm → ELU
              → AvgPool → Dropout2d
    Classifier : AdaptiveAvgPool2d(1,1) → Flatten → Linear

    The AdaptiveAvgPool2d in the classifier makes the output dimension
    independent of n_samples, so the model accepts any temporal length
    at inference time without modification.

    Parameters
    ----------
    n_channels   : int  — number of EEG channels (default: 64)
    n_samples    : int  — number of time samples per trial (default: 250)
    n_classes    : int  — number of output classes (default: 3)
    dropout_rate : float — dropout probability used in both blocks
    """

    def __init__(
        self,
        n_channels=64,
        n_samples=250,
        n_classes=3,
        dropout_rate=0.5):

        super().__init__()

        F1 = 8
        D = 2
        F2 = F1 * D


        # =================================================
        # BLOCK 1 — Temporal + Spatial convolutions
        # =================================================

        self.block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False),

            nn.BatchNorm2d(F1),

            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False),

            nn.BatchNorm2d(F1 * D),

            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 4)),

            nn.Dropout2d(p=dropout_rate))


        # =================================================
        # BLOCK 2 — Depthwise separable convolution
        # =================================================

        self.block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False),

            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False),

            nn.BatchNorm2d(F2),

            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 8)),

            nn.Dropout2d(p=dropout_rate))


        # =================================================
        # CLASSIFIER
        # AdaptiveAvgPool2d collapses (B, F2, 1, T') → (B, F2, 1, 1)
        # regardless of n_samples, making the Linear always F2-dimensional.
        # =================================================

        self.gap        = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, n_classes)


    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return self.classifier(x)


# =========================================================
# lightweightEEGNet — Lightweight architecture
# =========================================================

class lightweightEEGNet(nn.Module):
    """
    Lightweight depthwise-separable CNN for EEG decoding.

    Architecture
    ------------
    Temporal Conv2d (1→6, kernel 1×32)
    → Spatial depthwise Conv2d (6→24, kernel C×1)
    → BatchNorm → ELU → AvgPool(1×8) → Dropout(0.25)      [EPD1]
    → Depthwise separable Conv2d (24→24, kernel 1×16)
    → ELU → AvgPool(1×12) → Dropout(0.5)                   [EPD2]
    → Flatten → Linear(flatten_dim, n_classes)

    The flatten dimension is inferred automatically via a dummy
    forward pass so the classifier adapts to any n_samples value.

    Parameters
    ----------
    n_channels : int  — number of EEG channels (default: 64)
    n_samples  : int  — number of time samples per trial (default: 250)
    n_classes  : int  — number of output classes (default: 3)
    """

    def __init__(
        self,
        n_channels=64,
        n_samples=250,
        n_classes=3):

        super().__init__()


        # =================================================
        # TEMPORAL CONVOLUTION
        # =================================================

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(1, 32),
            padding=(0, 15),
            bias=False)


        # =================================================
        # SPATIAL CONVOLUTION (depthwise)
        # =================================================

        self.spatial_conv = nn.Conv2d(
            in_channels=6,
            out_channels=24,
            kernel_size=(n_channels, 1),
            groups=6,
            bias=False)


        # =================================================
        # BATCH NORMALIZATION
        # =================================================

        self.bn = nn.BatchNorm2d(24)


        # =================================================
        # EPD1 — ELU + Pooling + Dropout
        # =================================================

        self.epd1 = nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25))


        # =================================================
        # SEPARABLE CONVOLUTION
        # =================================================

        self.separable_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=24,
                out_channels=24,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=24,
                bias=False),

            nn.Conv2d(
                in_channels=24,
                out_channels=24,
                kernel_size=1,
                bias=False))


        # =================================================
        # EPD2 — ELU + Pooling + Dropout
        # =================================================

        self.epd2 = nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 12)),
            nn.Dropout(0.5))


        # =================================================
        # INFER FLATTEN DIMENSION AUTOMATICALLY
        # =================================================

        with torch.no_grad():

            dummy = torch.zeros(1, 1, n_channels, n_samples)

            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.bn(x)
            x = self.epd1(x)
            x = self.separable_conv(x)
            x = self.epd2(x)

            flatten_dim = x.view(1, -1).shape[1]


        # =================================================
        # CLASSIFIER
        # =================================================

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, n_classes))


    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.epd1(x)
        x = self.separable_conv(x)
        x = self.epd2(x)
        x = self.classifier(x)

        return x