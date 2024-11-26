import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pytorch3d.ops import sample_farthest_points, knn_gather, knn_points, ball_query
from generator import Generator
from discriminator import Discriminator
import random


# Utility for handling empty indices in point cloud grouping
def replace_empty_indices(indices: torch.Tensor) -> torch.Tensor:
    B, G, K = indices.shape
    mask = indices == -1
    first_index = indices[:, :, 0].unsqueeze(-1).expand(-1, -1, K)
    indices[mask] = first_index[mask]
    return indices


# Core Modules for Point Cloud Grouping and Tokenization
class PointcloudGrouping(nn.Module):
    def __init__(
        self, num_groups: int, group_size: int, group_radius: Optional[float] = None
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.group_radius = group_radius

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centers, _ = sample_farthest_points(
            points[:, :, :3], K=self.num_groups, random_start_point=True
        )
        if self.group_radius is None:
            _, indices, _ = knn_points(
                centers, points[:, :, :3], K=self.group_size, return_sorted=False
            )
            grouped_points = knn_gather(points, indices)
        else:
            _, indices, _ = ball_query(
                centers, points[:, :, :3], K=self.group_size, radius=self.group_radius
            )
            grouped_points = knn_gather(points, replace_empty_indices(indices))
        grouped_points[:, :, :, :3] -= centers.unsqueeze(2)
        if self.group_radius is not None:
            grouped_points /= self.group_radius
        return grouped_points, centers


class MiniPointNet(nn.Module):
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, 1),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, 1),
        )

    def forward(self, points) -> torch.Tensor:
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


# Positional Encoding Factory for ALiBi and Absolute Embeddings
class PositionalEncodingFactory(nn.Module):
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


# Block class for Encoder Blocks
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop_rate=0.0,
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
        x = x + self.attn(self.norm1(x), pos_bias)
        return x + self.mlp(self.norm2(x))


class PointRTDEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        token_dim,
        hidden_dim,
        num_heads,
        num_layers,
        corruption_ratio=0.8,
        noise_scale=1.0,
        pos_embedding_strategy="alibi",
        finetune=False,
    ):
        super().__init__()
        self.tokenizer = PointcloudTokenizer(
            num_groups=64,
            group_size=32,
            group_radius=None,
            token_dim=token_dim,
            input_channels=input_dim,
        )
        self.pos_encoding_factory = PositionalEncodingFactory(
            strategy=pos_embedding_strategy, encoder_dim=token_dim, num_heads=num_heads
        )

        # Initialize Discriminator and Generator models
        self.generator = Generator(
            token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )
        self.discriminator = Discriminator(
            token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )

        # Transformer encoder blocks for contextual encoding
        self.encoder_blocks = nn.ModuleList(
            [Block(token_dim, num_heads) for _ in range(num_layers)]
        )

        # CLS token for summarizing representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, mean=0, std=0.02)

        self.corruption_ratio = corruption_ratio
        self.noise_scale = noise_scale
        self.finetune = finetune  # Store the finetune flag

    # Create noisy tokens by multiplicative Gaussian noise
    def create_noisy_tokens(
        self, tokens: torch.Tensor, corruption_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create noisy versions of the tokens by corrupting a fraction of them with multiplicative noise.
        Args:
            tokens (torch.Tensor): Original token embeddings, shape (B, G, token_dim).
            corruption_ratio (float): Fraction of tokens to corrupt (0.0 to 1.0).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Noisy tokens and corruption mask.
        """
        B, G, C = tokens.shape
        noisy_tokens = tokens.clone()
        corrupted_mask = torch.ones(B, G, dtype=torch.bool, device=tokens.device)

        # Number of tokens to corrupt per batch element
        num_corrupt_tokens = int(G * corruption_ratio)

        if num_corrupt_tokens > 0:
            for i in range(B):
                # Randomly select indices to corrupt
                noisy_indices = random.sample(range(G), num_corrupt_tokens)
                corrupted_mask[i, noisy_indices] = False

                # Generate Gaussian noise centered at 0 for meaningful scaling
                scaling_factors = torch.normal(
                    mean=0.0,
                    std=self.noise_scale,
                    size=(len(noisy_indices), C),
                    device=tokens.device,
                )

                # Apply multiplicative noise to the selected tokens
                noisy_tokens[i, noisy_indices] *= scaling_factors

        return noisy_tokens, corrupted_mask

    def forward(self, points: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass of the PointRTDEncoder.
        Args:
            points (torch.Tensor): Input point cloud data, shape (B, N, input_dim).
        Returns:
            Tuple containing:
                - encoded_tokens: Contextually encoded tokens (B, G+1, token_dim)
                - cleaned_tokens: Denoised tokens from the generator (B, G+1, token_dim)
                - original_tokens: Tokens pre-corruption (B, G, token_dim)
                - corrupted_mask: Real/corrupt ground truth mask (B, G)
                - discriminator_output: Discriminator predictions (B, G, 1)
                - patches: Original patches for loss comparison (B, G, group_size, input_dim)
                - centers: Original patches' centers to reconstruct original pcd for visualization (B, G, 3)
        """
        # Tokenization and grouping
        tokens, centers, patches = self.tokenizer(points)

        # Dynamically determine corruption ratio
        corruption_ratio = (
            0.0 if self.finetune or not self.training else self.corruption_ratio
        )

        # Create noisy tokens and corrupted mask
        noisy_tokens, corrupted_mask = self.create_noisy_tokens(
            tokens, corruption_ratio
        )

        # Discriminator's predictions on noisy tokens
        discriminator_output = self.discriminator(noisy_tokens)

        # Boolean mask for uncorrupted tokens based on discriminator output
        is_uncorrupted = discriminator_output.squeeze(-1) >= 0.5

        # Select appropriate mask for the generator
        generator_mask = (
            corrupted_mask if self.finetune or not self.training else is_uncorrupted
        )

        # Pass noisy tokens and the selected mask to generator
        cleaned_tokens = self.generator(noisy_tokens, generator_mask)

        # Add positional encoding for contextual encoding
        pos_bias = self.pos_encoding_factory(centers)

        # Append CLS token to cleaned tokens and tokens
        batch_size = tokens.shape[0]
        cls_token_expanded = self.cls_token.expand(
            batch_size, -1, -1
        )  # Shape: (B, 1, token_dim)
        cleaned_tokens = torch.cat(
            [cls_token_expanded, cleaned_tokens], dim=1
        )  # Shape: (B, G+1, token_dim)
        tokens = torch.cat(
            [cls_token_expanded, tokens], dim=1
        )  # Shape: (B, G+1, token_dim)

        # Contextual encoding with transformer encoder blocks
        encoded_tokens = cleaned_tokens.clone()  # Start with denoised tokens
        for block in self.encoder_blocks:
            encoded_tokens = block(encoded_tokens, pos_bias)

        return (
            encoded_tokens,  # Contextually encoded tokens for downstream tasks
            cleaned_tokens,  # Cleaned tokens from generator
            tokens,  # Original tokens pre-corruption
            corrupted_mask,  # Real/corrupt ground truth mask
            discriminator_output,  # Discriminator predictions
            patches,  # Original patches for loss comparison
            centers,  # Original patches' centers to reconstruct original pcd for visualization
        )


# Example usage
if __name__ == "__main__":
    batch_size, num_pts, input_dim, token_dim, hidden_dim, num_heads, num_layers = (
        10,
        2048,
        4,
        256,
        256,
        8,
        6,
    )
    points = torch.randn(batch_size, num_pts, input_dim)

    encoder = PointRTDEncoder(
        input_dim=input_dim,
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(points.device)

    (
        encoded_tokens,
        cleaned_tokens,
        original_tokens,
        corrupted_mask,
        discriminator_output,
        patches,
    ) = encoder(points)
    print(
        "Output token shape:", encoded_tokens.shape
    )  # Expected: (batch_size, num_groups, token_dim)
    print(
        "Original patches shape:", patches.shape
    )  # Expected: (batch_size, num_groups, group_size, input_dim)
