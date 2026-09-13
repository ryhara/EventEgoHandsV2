import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        dropout_prob=0.2,
        dropout_mode="dropout2d",
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]
        if dropout_prob > 0:
            if dropout_mode == "dropout":
                layers.append(nn.Dropout(dropout_prob))
            elif dropout_mode == "dropout2d":
                layers.append(nn.Dropout2d(dropout_prob))
            else:
                raise ValueError(f"Invalid dropout mode: {dropout_mode}")
        layers += [
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout_prob > 0:
            if dropout_mode == "dropout":
                layers.append(nn.Dropout(dropout_prob))
            elif dropout_mode == "dropout2d":
                layers.append(nn.Dropout2d(dropout_prob))
        layers.append(nn.ReLU(inplace=True))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels, out_channels, dropout_prob=0.2, dropout_mode="dropout2d"
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                dropout_prob=dropout_prob,
                dropout_mode=dropout_mode,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        dropout_prob=0.2,
        dropout_mode="dropout2d",
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
                dropout_prob=dropout_prob,
                dropout_mode=dropout_mode,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                dropout_prob=dropout_prob,
                dropout_mode=dropout_mode,
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        dropout_prob=0.2,
        dropout_mode="dropout2d",
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(
            n_channels, 64, dropout_prob=dropout_prob, dropout_mode=dropout_mode
        )
        self.down1 = Down(64, 128, dropout_prob=dropout_prob, dropout_mode=dropout_mode)
        self.down2 = Down(
            128, 256, dropout_prob=dropout_prob, dropout_mode=dropout_mode
        )
        self.down3 = Down(
            256, 512, dropout_prob=dropout_prob, dropout_mode=dropout_mode
        )
        factor = 2 if bilinear else 1
        self.down4 = Down(
            512, 1024 // factor, dropout_prob=dropout_prob, dropout_mode=dropout_mode
        )
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            dropout_prob=dropout_prob,
            dropout_mode=dropout_mode,
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            dropout_prob=dropout_prob,
            dropout_mode=dropout_mode,
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            dropout_prob=dropout_prob,
            dropout_mode=dropout_mode,
        )
        self.up4 = Up(
            128, 64, bilinear, dropout_prob=dropout_prob, dropout_mode=dropout_mode
        )
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


def load_unet(checkpoint_path, n_channels, n_classes, device="cpu"):
    """Build a UNet matching a segmentation checkpoint and load its weights.

    Dropout layers take up a slot in the `DoubleConv` `nn.Sequential`, so a model
    built without them cannot load a checkpoint trained with them (and the other
    way round). The layout is readable from the keys themselves - `double_conv`
    ends at index 4 without dropout and at index 5 with it - so the structure is
    recovered from the checkpoint instead of being configured. Only the presence
    of the layers matters here: the models are used frozen and in eval mode,
    where the dropout probability has no effect.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    state_dict = {
        k.replace("module.", "").replace("model.", ""): v for k, v in checkpoint.items()
    }

    has_dropout = "inc.double_conv.5.weight" in state_dict
    model = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        dropout_prob=0.2 if has_dropout else 0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=1)
    print(model)
    input = torch.randn(1, 3, 260, 346)
    output = model(input)
    print(output.size())
