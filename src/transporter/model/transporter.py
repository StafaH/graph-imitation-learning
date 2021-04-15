import torch
from torch import nn
from model.model_utils import spatial_softmax
#import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1,
                 padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # return self.lrelu(x)
        return torch.relu(x)


class FeatureExtractor(nn.Module):
    """Feature Extractor Phi"""

    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            # Input 3 x 128 x 128                                                 Outputs
            ConvBlock(in_channels, 32, kernel_size=(7, 7), stride=1, padding=3),  # 1 32 x 128 x 128
            ConvBlock(32, 32, kernel_size=(3, 3), stride=1),                      # 2 32 x 128 x 128
            ConvBlock(32, 64, kernel_size=(3, 3), stride=2),                      # 3 64 x 64 x 64
            ConvBlock(64, 64, kernel_size=(3, 3), stride=1),                      # 4 64 x 64 x 64
            ConvBlock(64, 128, kernel_size=(3, 3), stride=2),                     # 5 128 x 32 x 32
            ConvBlock(128, 128, kernel_size=(3, 3), stride=1),                    # 6 128 x 32 x 32
        )

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.
        Returns
        =======
        y: (N, C, H, K) tensor.
        """
        return self.net(x)


class KeyNet(nn.Module):
    """KeyNet Psi"""

    def __init__(self, in_channels=3, k=1):
        super(KeyNet, self).__init__()
        self.net = nn.Sequential(
            # Input: 3 x 128 x 128                                                  Outputs
            ConvBlock(in_channels, 32, kernel_size=(7, 7), stride=1, padding=3),    # 1: 32 x 128 x 128
            ConvBlock(32, 32, kernel_size=(3, 3), stride=1),                        # 2: 32 x 128 x 128
            ConvBlock(32, 64, kernel_size=(3, 3), stride=2),                        # 3: 64 x 64 x 64
            ConvBlock(64, 64, kernel_size=(3, 3), stride=1),                        # 4: 64 x 64 x 64
            ConvBlock(64, 128, kernel_size=(3, 3), stride=2),                       # 5: 128 x 32 x 32
            ConvBlock(128, 128, kernel_size=(3, 3), stride=1),                      # 6: 128 x 32 x 32
        )
        self.regressor = nn.Conv2d(128, k, kernel_size=(1, 1))                      # 7: 5 x 32 x 32

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, k, H', W') tensor.
        """
        x = self.net(x)
        return self.regressor(x)


class RefineNet(nn.Module):
    """Network that generates images from feature maps and heatmaps."""

    def __init__(self, num_channels):
        super(RefineNet, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(128, 128, kernel_size=(3, 3), stride=1),                      # 1: 128 x 32 x 32
            ConvBlock(128, 64, kernel_size=(3, 3), stride=1),                       # 2: 64 x 32 x 32
            nn.UpsamplingBilinear2d(scale_factor=2),                                # 3: 64 x 64 x 64
            ConvBlock(64, 64, kernel_size=(3, 3), stride=1),                        # 4: 64 x 64 x 64
            ConvBlock(64, 32, kernel_size=(3, 3), stride=1),                        # 5: 32 x 64 x 64
            nn.UpsamplingBilinear2d(scale_factor=2),                                # 6: 32 x 128 x 128
            ConvBlock(32, 32, kernel_size=(3, 3), stride=1),                        # 2: 32 x 128 x 128
            ConvBlock(32, num_channels, kernel_size=(7, 7), stride=1, padding=3),   # 2: 3 x 128 x 128
        )

    def forward(self, x):
        """
        x: the transported feature map.
        """
        return self.net(x)


def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H
    S_col = features.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    # N, K
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    return torch.stack((u_row, u_col), -1)  # N, K, 2


def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    width, height = features.size(-1), features.size(-2)
    mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    inv_std = 1 / std
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std**2
    g_yx = torch.exp(-dist)
    # g_yx = g_yx.permute([0, 2, 3, 1])
    return g_yx


def transport(source_keypoints, target_keypoints, source_features,
              target_features):
    """
    Args
    ====
    source_keypoints (N, K, H, W)
    target_keypoints (N, K, H, W)
    source_features (N, D, H, W)
    target_features (N, D, H, W)
    Returns
    =======
    """
    out = source_features
    for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
    return out


class Transporter(nn.Module):

    def __init__(self, feature_extractor, key_net, refine_net, std=0.1):
        super(Transporter, self).__init__()
        self.feature_extractor = feature_extractor
        self.key_net = key_net
        self.refine_net = refine_net
        self.std = std

    def forward(self, source_images, target_images):
        source_features = self.feature_extractor(source_images)
        target_features = self.feature_extractor(target_images)

        source_keypoints = gaussian_map(
            spatial_softmax(self.key_net(source_images)), std=self.std)

        target_keypoints = gaussian_map(
            spatial_softmax(self.key_net(target_images)), std=self.std)

        transported_features = transport(source_keypoints.detach(),
                                         target_keypoints,
                                         source_features.detach(),
                                         target_features)

        assert transported_features.shape == target_features.shape

        reconstruction = self.refine_net(transported_features)
        return reconstruction


# Special Class for PyTorch Lightning Cluster Training
# class LitTransporter(pl.LightningModule):

#     def __init__(self, num_channels, num_keypoints, std=0.1):
#         super().__init__()
#         self.feature_extractor = FeatureExtractor(num_channels)
#         self.key_net = KeyNet(num_channels, num_keypoints)
#         self.refine_net = RefineNet(num_channels)
#         self.std = std

#     def forward(self, source_images, target_images):
#         source_features = self.feature_extractor(source_images)
#         target_features = self.feature_extractor(target_images)

#         source_keypoints = gaussian_map(
#             spatial_softmax(self.key_net(source_images)), std=self.std)

#         target_keypoints = gaussian_map(
#             spatial_softmax(self.key_net(target_images)), std=self.std)

#         transported_features = transport(source_keypoints.detach(),
#                                          target_keypoints,
#                                          source_features.detach(),
#                                          target_features)

#         assert transported_features.shape == target_features.shape

#         reconstruction = self.refine_net(transported_features)
#         return reconstruction

#     def training_step(self, batch, batch_idx):
#         source, target = batch
#         reconstruction = self(source, target)
#         loss = torch.nn.functional.mse_loss(reconstruction, target)
#         self.log('train_loss', loss, on_epoch=True)
#         return loss

#     def configure_optimizers(self):
#         # self.hparams available because we called self.save_hyperparameters()
#         optimizer = torch.optim.Adam(self.parameters(), 1e-3)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(25000), gamma=0.95)
#         return [optimizer], [scheduler]
