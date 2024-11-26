import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PointMAEDecoder(nn.Module):
    """Decoder for PointMAE to reconstruct masked point cloud tokens."""

    def __init__(
        self,
        decoder_dim: int,
        num_patches: int,
        num_pts_per_patch: int,
        num_channels: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        """
        Initializes the decoder.
        Args:
            decoder_dim (int): Dimension of each token in the decoder.
            num_patches (int): Number of patches in the point cloud.
            num_pts_per_patch (int): Number of points within each patch.
            num_channels (int): Number of channels per point (default is 3 for XYZ).
            num_layers (int): Number of transformer layers in the decoder.
            num_heads (int): Number of attention heads in each transformer layer.
            mlp_ratio (float): MLP expansion ratio.
            drop_rate (float): Dropout rate.
        """
        super(PointMAEDecoder, self).__init__()
        self.num_patches = num_patches
        self.num_pts_per_patch = num_pts_per_patch
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(decoder_dim, num_heads, mlp_ratio, drop_rate)
                for _ in range(num_layers)
            ]
        )

        # Adjust output projection to expand each patch embedding to individual points
        self.output_projection = nn.Linear(
            decoder_dim, num_pts_per_patch * num_channels
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            tokens (torch.Tensor): Input tensor of shape (B, num_patches + 1, C), where:
                - B = batch size
                - num_patches + 1 = number of patches + CLS token
                - C = token dimension
        Returns:
            torch.Tensor: Reconstructed point cloud of shape (B, num_patches, num_pts_per_patch, num_channels).
        """
        # Remove CLS token before decoding
        tokens = tokens[:, 1:, :]  # Shape: (B, num_patches, C)

        # Pass tokens through each decoder layer
        for layer in self.decoder_layers:
            tokens = layer(tokens)

        # Project to (num_pts_per_patch * num_channels) for each patch, then reshape
        tokens = self.output_projection(
            tokens
        )  # Shape: (B, num_patches, num_pts_per_patch * num_channels)
        return tokens.view(
            tokens.size(0), self.num_patches, self.num_pts_per_patch, self.num_channels
        )  # Shape: (B, num_patches, num_pts_per_patch, num_channels)


class DecoderBlock(nn.Module):
    """A single Transformer block for the PointMAE decoder."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1
    ):
        """
        Initializes a single decoder block.
        Args:
            dim (int): Dimension of input tokens.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): MLP expansion ratio.
            drop (float): Dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True
        )
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single decoder block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        x = x + self.drop_path(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Example usage of PointMAEDecoder
if __name__ == "__main__":
    batch_size, num_patches, num_pts_per_patch, decoder_dim, num_channels = (
        10,
        64,
        32,
        256,
        4,
    )
    encoded_tokens = torch.randn(
        batch_size, num_patches + 1, decoder_dim
    )  # Includes CLS token

    decoder = PointMAEDecoder(
        decoder_dim=decoder_dim,
        num_patches=num_patches,
        num_pts_per_patch=num_pts_per_patch,
        num_channels=num_channels,
    )
    reconstructed_points = decoder(encoded_tokens)

    print(
        "Reconstructed points shape:", reconstructed_points.shape
    )  # Expected: (B, num_patches, num_pts_per_patch, num_channels)
