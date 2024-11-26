import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from pytorch3d.ops import ball_query, knn_gather, knn_points, sample_farthest_points


# Utility function for handling empty indices in point cloud grouping
def replace_empty_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Replace empty indices (-1) with the first valid index within each group.
    Args:
        indices (torch.Tensor): Tensor of shape (B, G, K) where:
            - B = batch size
            - G = number of groups
            - K = group size
    Returns:
        torch.Tensor: Updated indices with no -1 values, same shape as input (B, G, K).
    """
    B, G, K = indices.shape
    mask = indices == -1
    first_index = indices[:, :, 0].unsqueeze(-1).expand(-1, -1, K)
    indices[mask] = first_index[mask]
    return indices


# Core Modules


class PointcloudGrouping(nn.Module):
    """Groups points into clusters for feature extraction based on farthest sampling or radius search."""

    def __init__(
        self, num_groups: int, group_size: int, group_radius: Optional[float] = None
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.group_radius = group_radius

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Group points based on farthest sampling or radius search.
        Args:
            points (torch.Tensor): Input points of shape (B, N, C) where:
                - B = batch size
                - N = number of points
                - C = point dimension (e.g., 3 for XYZ coordinates)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Grouped points and centers.
                - grouped_points: shape (B, G, K, C), where G is the number of groups
                - centers: shape (B, G, 3) for group centers in XYZ
        """
        centers, _ = sample_farthest_points(
            points[:, :, :3], K=self.num_groups, random_start_point=True
        )
        # centers shape: (B, G, 3)

        if self.group_radius is None:
            _, indices, _ = knn_points(centers, points[:, :, :3], K=self.group_size)
            # indices shape: (B, G, K)
            grouped_points = knn_gather(points, indices)
            # grouped_points shape: (B, G, K, C)
        else:
            _, indices, _ = ball_query(
                centers, points[:, :, :3], K=self.group_size, radius=self.group_radius
            )
            grouped_points = knn_gather(points, replace_empty_indices(indices))
            # grouped_points shape: (B, G, K, C)

        # Adjust points by subtracting group centers
        grouped_points[:, :, :, :3] -= centers.unsqueeze(2)
        if self.group_radius is not None:
            grouped_points /= self.group_radius
        return grouped_points, centers


class MiniPointNet(nn.Module):
    """Processes point cloud data to generate embeddings for each group of points."""

    def __init__(self, input_channels: int, output_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, 1),
        )

    def forward(self, points) -> torch.Tensor:
        """
        Generate point embeddings.
        Args:
            points (torch.Tensor): Tensor of shape (B, N, C), where C is the input channel size.
        Returns:
            torch.Tensor: Global feature embedding of shape (B, output_dim).
        """
        feature = self.first_conv(points.transpose(2, 1))  # (B, 256, N)
        global_feature = torch.max(feature, dim=2, keepdim=True).values  # (B, 256, 1)
        combined_feature = torch.cat(
            [global_feature.expand(-1, -1, feature.shape[2]), feature], dim=1
        )  # (B, 512, N)
        return torch.max(
            self.second_conv(combined_feature), dim=2
        ).values  # (B, output_dim)


class PointcloudTokenizer(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_size: int,
        group_radius: Optional[float],
        token_dim: int,
        input_channels: int,
    ):
        super().__init__()
        self.grouping = PointcloudGrouping(num_groups, group_size, group_radius)
        self.embedding = MiniPointNet(input_channels, token_dim)

    def forward(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups, centers = self.grouping(
            points
        )  # groups: (B, G, K, C), centers: (B, G, 3)
        B, G, K, C = groups.shape
        tokens = self.embedding(groups.reshape(B * G, K, C)).reshape(B, G, -1)
        return tokens, centers, groups  # Return tokens, centers, and original patches


class PositionalEncodingFactory(nn.Module):
    """Factory to select and return the specified positional encoding."""

    def __init__(self, strategy: str, encoder_dim: int, num_heads: int):
        super().__init__()
        if strategy == "absolute":
            self.encoding = nn.Sequential(
                nn.Linear(3, 128), nn.GELU(), nn.Linear(128, encoder_dim)
            )
        elif strategy == "alibi":
            slopes = torch.tensor([2 ** -(2 ** -(i + 3)) for i in range(num_heads)])
            self.encoding = lambda centers: -torch.cdist(
                centers, centers, p=2
            ).unsqueeze(1) * slopes.view(1, num_heads, 1, 1).to(centers.device)
        else:
            raise ValueError(f"Unknown positional embedding strategy: {strategy}")

    def forward(self, centers: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Generate positional encoding.
        Args:
            centers (torch.Tensor): Tensor of shape (B, G, 3), where G is the number of groups.
        Returns:
            torch.Tensor: Positional encoding of shape (B, num_heads, G, G).
        """
        return self.encoding(centers)


class Attention(nn.Module):
    """Self-attention mechanism with optional ALiBi bias for long-sequence handling."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Linear layer for Q, K, V
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, pos_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) where:
                - B = batch size
                - N = number of tokens (including optional CLS token)
                - C = input dimension
            pos_bias: Positional bias tensor of shape (B, num_heads, G, G) where:
                - G = number of groups (or tokens) before adding CLS token
        """
        B, N, C = x.shape  # (B, N, C)

        # Step 1: Project x to get Q, K, V matrices
        qkv = self.qkv(x)  # (B, N, 3 * C), where C = dim
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        # qkv is reshaped to (3, B, num_heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v each have shape (B, num_heads, N, head_dim)

        # Step 2: Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn shape: (B, num_heads, N, N) after matrix multiplication and scaling

        # Step 3: Add positional bias if provided
        if pos_bias is not None:
            # Adjust pos_bias if CLS token is added (CLS token increases N by 1)
            if pos_bias.size(-1) == N - 1:
                pos_bias = F.pad(pos_bias, (1, 0, 1, 0))
            # pos_bias now has shape (B, num_heads, N, N) after padding, compatible with attn
            attn += pos_bias

        # Step 4: Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)  # (B, num_heads, N, N)
        attn = self.attn_drop(attn)  # Dropout after softmax

        # Step 5: Multiply attention weights with values (V)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Output x shape: (B, N, C), where C = dim

        # Step 6: Apply final output projection
        x = self.proj(x)  # (B, N, C)
        x = self.proj_drop(x)  # Dropout on the projected output
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP layers."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop_rate=0.0,
        use_alibi=False,
    ):
        super().__init__()
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_rate, drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x, pos_bias=None):
        """
        Apply self-attention and MLP layer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            pos_bias (Optional[torch.Tensor]): Positional bias for attention, shape (B, num_heads, N, N).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        x = x + self.attn(self.norm1(x), pos_bias)
        return x + self.mlp(self.norm2(x))


# Main Encoder Class without Classification Head


class PointMAEEncoder(nn.Module):
    """Point cloud encoder with tokenization, positional encoding, and Transformer blocks."""

    def __init__(
        self,
        input_dim,
        dropout=0.1,
        num_groups=64,
        group_size=32,
        group_radius=None,
        num_heads=8,
        num_layers=6,
        encoder_dim=256,
        pos_embedding_strategy="alibi",
        mask_ratio=0.65,
        masking_strategy="learned",
        finetune=False,  # Add finetune flag to disable masking
    ):
        super().__init__()
        self.tokenizer = PointcloudTokenizer(
            num_groups, group_size, group_radius, encoder_dim, input_dim
        )
        self.pos_encoding_factory = PositionalEncodingFactory(
            pos_embedding_strategy, encoder_dim, num_heads
        )
        self.encoder = nn.ModuleList(
            [
                Block(encoder_dim, num_heads, drop=dropout, attn_drop_rate=0.0)
                for _ in range(num_layers)
            ]
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, encoder_dim)
        )  # Initialized in __init__
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.finetune = finetune  # Flag to disable masking in fine-tuning
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        nn.init.trunc_normal_(self.cls_token, mean=0, std=0.02)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Process point cloud tokens through the encoder and apply masking.
        Args:
            points (torch.Tensor): Input points of shape (B, N, C).
        Returns:
            torch.Tensor: Encoded shape embeddings, shape (B, N, encoder_dim).
        """
        tokens, centers, patches = self.tokenizer(points)
        pos_bias = self.pos_encoding_factory(centers)

        # Append CLS token if required by pooling strategy
        cls_token_expanded = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token_expanded, tokens], dim=1)

        # Forward pass through the encoder
        for block in self.encoder:
            tokens = block(tokens, pos_bias)

        # Masking strategy applied AFTER encoder
        if (
            not self.finetune
            and self.training
            and self.masking_strategy in ["zeros", "learned"]
        ):
            B, T, C = tokens.shape
            num_masked = int(self.mask_ratio * T)
            rand_indices = torch.rand(B, T).argsort(dim=1)
            mask_indices = rand_indices[:, :num_masked]

            if self.masking_strategy == "learned":
                tokens[torch.arange(B).unsqueeze(1), mask_indices] = self.mask_token
            elif self.masking_strategy == "zeros":
                tokens[torch.arange(B).unsqueeze(1), mask_indices] = 0

        return tokens, patches  # Final token embeddings for all input points


# Example usage

if __name__ == "__main__":
    batch_size, num_pts, channels = 10, 2048, 4
    point_clouds = torch.randn(batch_size, num_pts, channels)
    encoder = PointMAEEncoder(input_dim=channels, dropout=0.1)
    shape_embedding, original_patches = encoder(point_clouds)
    print(
        "Output shape:", shape_embedding.shape
    )  # Expected: (B, num_patches + 1, encoder_dim)
    print(
        "Original patches shape:", original_patches.shape
    )  # Expected: (B, num_patches, num_pts_per_patch, pt_features)
