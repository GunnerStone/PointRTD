import torch
import torch.nn as nn
from pointRTDEncoder import PointRTDEncoder
from pointMAEDecoder import PointMAEDecoder
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


class PointRTDModel(nn.Module):
    def __init__(
        self,
        input_dim,
        token_dim,  # Adjusted to match PointMAEDecoder
        hidden_dim,
        num_heads,
        num_layers,
        corruption_ratio=0.8,
        noise_scale=0.9,
        num_patches=64,
        num_pts_per_patch=32,
        finetune=False,
    ):
        super(PointRTDModel, self).__init__()
        self.encoder = PointRTDEncoder(
            input_dim=input_dim,
            token_dim=token_dim,  # Matches decoder_dim in PointMAEDecoder
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            corruption_ratio=corruption_ratio,
            noise_scale=noise_scale,
            finetune=finetune,
        )
        self.decoder = PointMAEDecoder(
            decoder_dim=token_dim,  # Use decoder_dim instead of token_dim
            num_patches=num_patches,
            num_pts_per_patch=num_pts_per_patch,
            num_channels=input_dim,  # Matches input dimensions of the original points
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def forward(self, pointclouds):
        """
        Forward pass through the RTD model, which includes the encoder and decoder.
        Args:
            pointclouds (torch.Tensor): Input batch of point clouds, shape (B, N, input_dim).
        Returns:
            Tuple of relevant vectors for loss calculation.
        """
        (
            encoded_tokens,  # Encoded tokens from the encoder (contextual)
            cleaned_tokens,  # Cleaned tokens from the generator
            original_tokens,  # Original tokens pre-corruption
            corrupted_mask,  # Real/corrupt ground truth mask
            discriminator_output,  # Discriminator predictions
            patches,  # Original patches for reconstructive loss
            centers,  # Original patches' centers to reconstruct original pcd for visualization
        ) = self.encoder(pointclouds)

        # Pass relevant vectors through the decoder for reconstruction
        reconstructed_patches = self.decoder(encoded_tokens)

        return (
            reconstructed_patches,
            patches,
            corrupted_mask,
            discriminator_output,
            original_tokens,
            cleaned_tokens,
            centers,  # Original patches' centers to reconstruct original pcd for visualization
        )

    def get_loss(
        self,
        reconstructed_patches,
        patches,
        corrupted_mask,
        discriminator_output,
        original_tokens,
        cleaned_tokens,
        centers,
    ):
        """
        Calculate the losses for reconstruction, discriminator, and generator.
        Args:
            reconstructed_patches: Output from decoder, reconstructed patches.
            patches: Original point cloud patches.
            corrupted_mask: Ground truth mask for real/corrupt tokens.
            discriminator_output: Discriminator’s output for real/corrupt classification.
            original_tokens: Tokens pre-corruption.
            cleaned_tokens: Cleaned tokens from the generator.
            centers: center coordinates of patches pre-normalization

        Returns:
            dict of losses: Contains reconstruction, discriminator, and generator loss values.
        """
        batch_size = patches.shape[0]
        num_patches = patches.shape[1]  # Second dimension corresponds to num_patches
        num_channels = patches.shape[3]
        # Reconstruction Loss (MSE between decoder output and original patches)
        # Initialize a variable to accumulate the reconstruction loss across patches
        reconstruction_loss = 0.0

        # Iterate over each patch and calculate Chamfer Distance independently
        for i in range(num_patches):
            reconstructed_patch = reconstructed_patches[
                :, i, :, :
            ]  # Shape: (batch_size, num_pts_per_patch, num_channels)
            original_patch = patches[
                :, i, :, :
            ]  # Shape: (batch_size, num_pts_per_patch, num_channels)

            # Compute Chamfer Distance for the current patch
            patch_loss, _ = chamfer_distance(reconstructed_patch, original_patch)

            # Accumulate loss
            reconstruction_loss += patch_loss

        # Average the reconstruction loss across all patches
        reconstruction_loss /= num_patches

        # 2. Unnormalize Patches to Reconstruct Full Point Clouds
        # Add centers back to undo normalization
        patches_unnormalized = (
            patches.clone()
        )  # Shape: (B, num_patches, num_points_per_patch, 3)
        reconstructed_patches_unnormalized = (
            reconstructed_patches.clone()
        )  # Same shape as patches

        # Add the corresponding centers back to each patch
        patches_unnormalized[:, :, :, :3] += centers.unsqueeze(
            2
        )  # Add XYZ centers to patches
        reconstructed_patches_unnormalized[:, :, :, :3] += centers.unsqueeze(2)

        # 3. Combine All Patches into Complete Point Clouds
        # Reshape from (B, num_patches, num_points_per_patch, 3) -> (B, num_patches * num_points_per_patch, 3)
        original_full_pcd = patches_unnormalized.reshape(
            batch_size, -1, 3
        )  # Shape: (B, num_pts, 3)
        reconstructed_full_pcd = reconstructed_patches_unnormalized.reshape(
            batch_size, -1, 3
        )  # Shape: (B, num_pts, 3)

        # 4. Compute Chamfer Distance Between Full Point Clouds for the Entire Batch
        # Chamfer distance expects input of shape (B, P, D)
        full_pcd_loss, _ = chamfer_distance(
            reconstructed_full_pcd, original_full_pcd
        )  # Full batch loss

        # Discriminator Loss (Binary Cross-Entropy)
        real_labels = (
            corrupted_mask.float()
        )  # Ground truth (1 for real, 0 for corrupted)
        # Calculate class weights
        num_real = torch.sum(real_labels)  # Number of clean (real) tokens
        num_fake = real_labels.numel() - num_real  # Number of corrupted (fake) tokens

        # Avoid division by zero by ensuring num_fake > 0
        weight_fake = 1.0 / (num_fake + 1e-6)  # Weight for fake tokens
        weight_real = 1.0 / (num_real + 1e-6)  # Weight for real tokens

        # Create weight tensor for each token
        weights = torch.where(real_labels == 1, weight_real, weight_fake)

        # Compute weighted binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(
            discriminator_output,
            real_labels.unsqueeze(-1),
            weight=weights.unsqueeze(-1),
        )

        # Add a confidence penality to enforce stronger predictions
        confidence_penalty = torch.mean(
            (discriminator_output - real_labels.unsqueeze(-1)) ** 2
        )
        lambda_penalty = 1.0  # Hyperparameter to balance BCE and confidence penalty
        discriminator_loss = bce_loss + lambda_penalty * confidence_penalty

        # Generator Loss (MSE between cleaned tokens and original tokens)
        # Exclude the CLS token by selecting tokens starting from index 1
        original_tokens_no_cls = original_tokens[:, 1:, :]  # Shape: (32, 64, 256)
        cleaned_tokens_no_cls = cleaned_tokens[:, 1:, :]  # Shape: (32, 64, 256)

        # Select only the corrupted tokens for loss calculation
        fake_tokens = original_tokens_no_cls[~corrupted_mask]
        reconstructed_fake_tokens = cleaned_tokens_no_cls[~corrupted_mask]

        # Compute MSE loss only for the corrupted tokens
        if fake_tokens.numel() > 0:  # Check if there are any fake tokens
            generator_loss = F.mse_loss(reconstructed_fake_tokens, fake_tokens)
        else:
            generator_loss = torch.tensor(
                0.0, device=original_tokens.device
            )  # Ensure it's a tensor  # Set to 0 if no corrupted tokens

        # Return a dictionary of losses
        return reconstruction_loss, discriminator_loss, generator_loss, full_pcd_loss


# Example usage
if __name__ == "__main__":
    # Settings
    batch_size, num_pts, input_dim, decoder_dim, hidden_dim, num_heads, num_layers = (
        10,
        2048,
        4,
        256,
        256,
        8,
        6,
    )

    # Generate random input point clouds
    pointclouds = torch.randn(batch_size, num_pts, input_dim)

    # Initialize PointRTDModel
    model = PointRTDModel(
        input_dim=input_dim,
        decoder_dim=decoder_dim,  # Updated to match PointMAEDecoder's decoder_dim
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Forward pass
    (
        reconstructed_patches,
        patches,
        corrupted_mask,
        discriminator_output,
        original_tokens,
        cleaned_tokens,
    ) = model(pointclouds)

    # Calculate losses
    losses = model.get_loss(
        reconstructed_patches,
        patches,
        corrupted_mask,
        discriminator_output,
        original_tokens,
        cleaned_tokens,
    )

    # Print loss values
    print("Reconstruction Loss:", losses["reconstruction_loss"].item())
    print("Discriminator Loss:", losses["discriminator_loss"].item())
    print("Generator Loss:", losses["generator_loss"].item())
