"""
TrackNetV2 PyTorch model — vendored from mareksubocz/TrackNet, adapted to
match the BatchNorm convention used in Andrew Dettor's pickleball-trained
SavedModel.

The TrackNet class below is reconstructed from the architecture described
in the TrackNetV2 paper (Sun et al., 2020) and the published source of
mareksubocz/TrackNet (https://github.com/mareksubocz/TrackNet, archived
Sep 2024). It is a 3-in-3-out heatmap-based small-object tracker:
input is 3 consecutive RGB frames stacked along channel axis (9 channels
total); output is 3 heatmaps, one per input frame, with values in [0, 1]
(after sigmoid) indicating ball-presence probability per pixel.

NOTE on BatchNorm:

Dettor's training code used Keras `BatchNormalization` layers with the
default `axis=-1` argument while feeding NCHW (channels-first) data.
With NCHW input, axis=-1 is the WIDTH axis, not the channels axis. So
his BN layers normalize over the rightmost spatial dimension instead
of channels. This is unconventional but it is the convention his
trained weights expect.

To stay faithful to those trained weights we cannot use the standard
`nn.BatchNorm2d` (which normalizes channels). Instead this file defines
`BatchNormOverWidth`: a custom 4-parameter normalization with parameter
size equal to the width dimension at the layer's position. The
TrackNet __init__ constructs each BN with the correct width parameter
for its position in the encoder / decoder.

The native input resolution is (288, 512). At each encoder downsample
the width halves (512 -> 256 -> 128 -> 64). In the decoder the width
doubles back up (64 -> 128 -> 256 -> 512). Each BatchNormOverWidth
instance has parameter size equal to the width at its position.

If TrackNet is ever used at a non-(288, 512) resolution, every BN will
need to be re-instantiated with the new widths. The model's __init__
takes an optional `input_shape` argument to support this — defaults to
(288, 512) which is what Dettor's weights were trained on.

------------------------------------------------------------------------
Original mareksubocz/TrackNet license:

MIT License

Copyright (c) 2023 Marek Subocz

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormOverWidth(nn.Module):
    """Per-width-position batch normalization.

    For input (N, C, H, W), normalizes over (N, C, H) — i.e. computes
    one mean/var per W position. Has 4 learnable/buffered parameters
    each of size W: gamma, beta, running_mean, running_var.

    This matches the behavior of Keras `BatchNormalization(axis=-1)`
    applied to channels-first data, which is what Dettor's training
    pipeline did.

    Forward pass uses running statistics (eval mode); we never train
    this model in our pipeline. Train-mode behavior is implemented for
    completeness but not exercised.
    """

    def __init__(self, num_features: int, eps: float = 1e-3,
                 momentum: float = 0.99):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))   # gamma
        self.bias = nn.Parameter(torch.zeros(num_features))    # beta
        # Running statistics (buffers, not parameters)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        if x.shape[-1] != self.num_features:
            raise RuntimeError(
                f"BatchNormOverWidth: input width {x.shape[-1]} does not "
                f"match num_features {self.num_features}. This BN was "
                f"constructed for a different position in the model."
            )
        if self.training:
            # Compute mean/var over (N, C, H) for each W position
            mean = x.mean(dim=(0, 1, 2))
            var = x.var(dim=(0, 1, 2), unbiased=False)
            # Update running stats (this branch is not used in our pipeline)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean.detach(), alpha=1.0 - self.momentum)
                self.running_var.mul_(self.momentum).add_(
                    var.detach(), alpha=1.0 - self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean = self.running_mean
            var = self.running_var
        # Apply: (x - mean) / sqrt(var + eps) * gamma + beta
        # Broadcast: mean/var/weight/bias are (W,), x is (N, C, H, W)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class TrackNet(nn.Module):
    """TrackNetV2: VGG16-encoder + decoder with skip connections, 3-in-3-out.

    Args:
        in_channels: number of input channels. Default 9 (3 frames * 3 RGB).
        out_channels: number of output heatmap channels. Default 3.
        dropout_rate: dropout applied inside each conv sublayer. Default 0.0.
        input_shape: (H, W) of the model's expected input. Default (288, 512).
            Used to construct BatchNormOverWidth instances with correct
            per-layer widths. If you change this, you must retrain — the
            BN parameter sizes change with width.

    Input shape: (batch, in_channels, H, W). Default H=288, W=512.
    Output shape: (batch, out_channels, H, W), values in [0, 1] after sigmoid.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 3,
                 dropout_rate: float = 0.0,
                 input_shape: tuple = (288, 512)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_h, self.input_w = input_shape

        # Compute width at each level of the encoder (after each maxpool)
        # and decoder (after each upsample). Pool reduces by 2; upsample
        # increases by 2.
        w0 = self.input_w           # 512  (full res, encoder block 1)
        w1 = w0 // 2                # 256  (after pool 1, encoder block 2)
        w2 = w1 // 2                # 128  (after pool 2, encoder block 3)
        w3 = w2 // 2                # 64   (after pool 3, encoder block 4 / bottleneck)
        # Decoder mirrors the encoder
        wd3 = w2                    # 128  (after upsample 1, decoder block 3)
        wd2 = w1                    # 256  (after upsample 2, decoder block 2)
        wd1 = w0                    # 512  (after upsample 3, decoder block 1)

        # ---- Encoder ----
        self.enc1 = self._make_conv_block(in_channels, 64, num=2,
                                          width=w0,
                                          dropout_rate=dropout_rate)
        self.enc2 = self._make_conv_block(64, 128, num=2,
                                          width=w1,
                                          dropout_rate=dropout_rate)
        self.enc3 = self._make_conv_block(128, 256, num=3,
                                          width=w2,
                                          dropout_rate=dropout_rate)
        self.enc4 = self._make_conv_block(256, 512, num=3,
                                          width=w3,
                                          dropout_rate=dropout_rate)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Decoder ----
        self.dec3 = self._make_conv_block(768, 256, num=3,
                                          width=wd3,
                                          dropout_rate=dropout_rate)
        self.dec2 = self._make_conv_block(384, 128, num=2,
                                          width=wd2,
                                          dropout_rate=dropout_rate)
        self.dec1 = self._make_conv_block(192, 64, num=2,
                                          width=wd1,
                                          dropout_rate=dropout_rate)

        # Final 1x1 conv (no BN follows it)
        self.head = nn.Conv2d(64, out_channels, kernel_size=(1, 1), padding=0)

    @staticmethod
    def _make_conv_sublayer(in_channels: int, out_channels: int,
                             width: int,
                             dropout_rate: float = 0.0) -> nn.Sequential:
        """Conv2d(3x3, same padding) -> ReLU -> BatchNormOverWidth -> [Dropout].

        Note: BatchNormOverWidth's parameter size is `width`, NOT
        `out_channels`, because Dettor's training applied BN over the
        width axis (Keras default axis=-1 with NCHW data).
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                      padding="same"),
            nn.ReLU(inplace=True),
            BatchNormOverWidth(num_features=width),
        ]
        if dropout_rate > 1e-15:
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _make_conv_block(self, in_channels: int, out_channels: int,
                         num: int, width: int,
                         dropout_rate: float = 0.0) -> nn.Sequential:
        """Stack of `num` conv sublayers, each followed by BN-over-width."""
        sublayers = [self._make_conv_sublayer(in_channels, out_channels,
                                               width=width,
                                               dropout_rate=dropout_rate)]
        for _ in range(num - 1):
            sublayers.append(self._make_conv_sublayer(
                out_channels, out_channels,
                width=width, dropout_rate=dropout_rate))
        return nn.Sequential(*sublayers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = F.interpolate(e4, scale_factor=2, mode="nearest")
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, scale_factor=2, mode="nearest")
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.head(d1)
        out = torch.sigmoid(out)
        return out


def count_layers(model: nn.Module) -> dict:
    """Diagnostic helper: count Conv2d and BatchNormOverWidth layers."""
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_bn = sum(1 for m in model.modules() if isinstance(m, BatchNormOverWidth))
    return {"conv": n_conv, "bn": n_bn}


if __name__ == "__main__":
    m = TrackNet(in_channels=9, out_channels=3)
    counts = count_layers(m)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TrackNet built: in=9, out=3, dropout=0.0, input=(288, 512)")
    print(f"layer counts: {counts}")
    print(f"total parameters: {n_params:,}")
    # Print BN sizes per layer to verify per-position widths
    print("BN layer widths (in forward order):")
    for name, mod in m.named_modules():
        if isinstance(mod, BatchNormOverWidth):
            print(f"  {name:30s}  num_features={mod.num_features}")
    x = torch.zeros(1, 9, 288, 512)
    with torch.no_grad():
        y = m(x)
    print(f"forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"output range: [{y.min().item():.4f}, {y.max().item():.4f}]")