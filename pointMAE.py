import torch
import torch.nn as nn
from pointMAEEncoder import PointMAEEncoder
from pointMAEDecoder import PointMAEDecoder
from pytorch3d.loss import chamfer_distance


class PointMAEModel(nn.Module):
    def __init__(
        self,
        input_dim,
        token_dim,
        num_heads,
        num_layers,
        mask_ratio=0.65,
        noise_scale=0.1,
        num_patches=64,
        num_pts_per_patch=32,
        finetune=False,  # Add finetune flag to disable masking
    ):
        super(PointMAEModel, self).__init__()
        self.encoder = PointMAEEncoder(
            input_dim=input_dim,
            encoder_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mask_ratio=mask_ratio,
            masking_strategy="learned",
            finetune=finetune,  # Add finetune flag to disable masking
        )
        self.decoder = PointMAEDecoder(
            decoder_dim=token_dim,
            num_patches=num_patches,
            num_pts_per_patch=num_pts_per_patch,
            num_channels=input_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def forward(self, pointclouds):
        """
        Forward pass through the MAE model, which includes the encoder and decoder.
        Args:
            pointclouds (torch.Tensor): Input batch of point clouds, shape (B, N, input_dim).
        Returns:
            Tuple containing reconstructed patches and original patches for loss calculation.
        """
        # Encode point clouds to tokens and generate original patches
        encoded_tokens, original_patches = self.encoder(pointclouds)

        # Decode to reconstruct point cloud patches
        reconstructed_patches = self.decoder(encoded_tokens)

        return reconstructed_patches, original_patches

    def get_loss(self, reconstructed_patches, original_patches):
        """
        Calculate the reconstruction loss using Chamfer Distance.
        Args:
            reconstructed_patches: Output from decoder, reconstructed patches.
            original_patches: Original point cloud patches.

        Returns:
            reconstruction_loss: Calculated loss value.
        """
        # Reconstruction Loss (Chamfer Distance)
        reconstruction_loss = 0.0
        num_patches = original_patches.shape[1]

        for i in range(num_patches):
            reconstructed_patch = reconstructed_patches[:, i, :, :]
            original_patch = original_patches[:, i, :, :]

            patch_loss, _ = chamfer_distance(reconstructed_patch, original_patch)
            reconstruction_loss += patch_loss

        reconstruction_loss /= num_patches  # Average loss over patches

        return reconstruction_loss


# Example usage
if __name__ == "__main__":
    batch_size, num_pts, input_dim, token_dim, num_heads, num_layers = (
        10,
        2048,
        3,
        256,
        8,
        6,
    )

    # Generate random input point clouds
    pointclouds = torch.randn(batch_size, num_pts, input_dim)

    # Initialize PointMAEModel
    model = PointMAEModel(
        input_dim=input_dim,
        token_dim=token_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Forward pass
    reconstructed_patches, original_patches = model(pointclouds)

    # Calculate loss
    loss = model.get_loss(reconstructed_patches, original_patches)
    print("Reconstruction Loss:", loss.item())
