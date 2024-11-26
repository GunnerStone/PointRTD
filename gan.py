import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator


class GAN(nn.Module):
    def __init__(self, token_dim, hidden_dim, num_layers, seq_len):
        super(GAN, self).__init__()
        self.generator = Generator(
            token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )
        self.discriminator = Discriminator(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            seq_len=seq_len,
        )

    def forward(self, original_tokens):
        """
        Arguments:
        - original_tokens: Tensor of shape (batch_size, seq_len, token_dim) -- the original sequence of tokens.

        Returns:
        - final_tokens: Sequence containing both real and denoised tokens (processed by the generator for fake ones).
        - discriminator_output: Binary predictions from the discriminator.
        """
        # Step 1: Discriminator classifies tokens as real or fake
        # Outputs undergo sigmoid activation before being returned as logits
        discriminator_output = self.discriminator(
            original_tokens
        )  # Shape: (batch_size, seq_len, 1)

        # Step 2: Prepare a mask for the fake tokens (discriminator_output < 0.5 is considered fake)
        real_fake_labels = (discriminator_output >= 0.5).squeeze(
            -1
        )  # Boolean mask for real tokens [batch_size, seq_len]

        # Step 3: Pass only the fake tokens to the generator for denoising
        denoised_tokens = self.generator(original_tokens, real_fake_labels)

        # Step 4: Create final token sequence where real tokens are unchanged and fake tokens are replaced by denoised ones
        final_tokens = torch.where(
            real_fake_labels.unsqueeze(-1).expand_as(original_tokens).clone(),
            original_tokens.clone(),
            denoised_tokens.clone(),
        )

        return final_tokens, discriminator_output


# Loss function for the Generator (pure MSE loss, since the goal is to denoise)
def generator_loss(generated_tokens, ground_truth_tokens):
    # L2 loss (MSE) for the generator to produce tokens similar to the ground truth (zeros in this case)
    loss_fn_mse = nn.MSELoss()
    loss_mse = loss_fn_mse(generated_tokens, ground_truth_tokens)
    return loss_mse


def discriminator_loss(real_discriminator_output, fake_discriminator_output):
    # Goal: Make the discriminator classify real tokens as real (1) and fake tokens as fake (0)
    real_loss_fn = nn.BCELoss()
    fake_loss_fn = nn.BCELoss()

    # Real tokens should be classified as real (1)
    real_target = torch.ones_like(real_discriminator_output)
    real_loss = real_loss_fn(real_discriminator_output, real_target)

    # Fake tokens should be classified as fake (0)
    fake_target = torch.zeros_like(fake_discriminator_output)
    fake_loss = fake_loss_fn(fake_discriminator_output, fake_target)

    # Total loss for the discriminator is the sum of real and fake losses
    return real_loss + fake_loss


# Helper function to create noisy tokens
def create_noisy_tokens(
    original_tokens, noise_ratio=0.8, noise_scale=0.1, device="cpu"
):
    """
    Create a noisy version of the input tokens by corrupting a fraction of them.

    Arguments:
    - original_tokens: The original (clean) token embeddings.
    - noise_ratio: The proportion of tokens to be corrupted (default is 0.8, or 80% noisy).
    - noise_scale: The standard deviation of the Gaussian noise to add (default is 0.1).
    - device: The device (cuda/cpu) where the operation is performed.

    Returns:
    - noisy_tokens: The noisy version of the original tokens.
    - clean_mask: Boolean mask indicating which tokens are clean (True) or noisy (False).
    """
    batch_size, seq_len, token_dim = original_tokens.shape
    noisy_tokens = original_tokens.clone()  # Start with a copy of the original tokens

    # Create a mask for clean (real) tokens on the correct device: True means clean, False means noisy
    clean_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # For each sequence, randomly corrupt a portion of the tokens
    num_noisy_tokens = int(seq_len * noise_ratio)
    for i in range(batch_size):
        noisy_indices = random.sample(range(seq_len), num_noisy_tokens)
        clean_mask[i, noisy_indices] = False  # Mark these tokens as noisy

        # Add noise to the noisy tokens (without in-place modification)
        noise = noise_scale * torch.randn(num_noisy_tokens, token_dim, device=device)
        noisy_tokens[i, noisy_indices] = noisy_tokens[i, noisy_indices] + noise

    return noisy_tokens, clean_mask


# Train the generator to denoise the fake tokens
if __name__ == "__main__":
    import random

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example settings
    batch_size = 4
    seq_len = 10  # Number of tokens per sequence
    token_dim = 128  # Dimensionality of each token (embedding size)
    hidden_dim = 256  # Hidden layer dimension
    num_layers = 2  # Number of transformer decoder layers

    # Initialize the GAN and move it to the GPU
    gan_model = GAN(
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
    ).to(device)

    # Create a batch of example clean input tokens (zeros as ground truth)
    original_tokens = torch.zeros(batch_size, seq_len, token_dim).to(
        device
    )  # Move to GPU

    # Introduce noise to a fraction of the tokens and move to the GPU
    noisy_tokens, clean_mask = create_noisy_tokens(
        original_tokens, noise_ratio=0.8, device=device
    )
    noisy_tokens, clean_mask = noisy_tokens.to(device), clean_mask.to(device)

    # Optimizers for generator and discriminator
    generator_optimizer = torch.optim.Adam(gan_model.generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(
        gan_model.discriminator.parameters(), lr=0.001
    )

    # Training loop for toy example
    for epoch in range(100):  # 100 epochs for a toy example
        # Zero the gradients
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # Forward pass through the GAN (discriminator + generator)
        final_tokens, discriminator_output = gan_model(noisy_tokens)

        # Split the discriminator output into real/fake outputs (for training discriminator)
        real_discriminator_output = discriminator_output[
            clean_mask
        ]  # Where tokens are clean
        fake_discriminator_output = discriminator_output[
            ~clean_mask
        ]  # Where tokens are noisy

        # Compute the discriminator loss
        d_loss = discriminator_loss(
            real_discriminator_output, fake_discriminator_output
        )
        d_loss.backward()  # Backpropagate for discriminator
        discriminator_optimizer.step()  # Update discriminator weights

        # Compute the generator loss (purely MSE to denoise the noisy tokens)
        g_loss = generator_loss(final_tokens, original_tokens)
        g_loss.backward()  # Backpropagate for generator
        generator_optimizer.step()  # Update generator weights

        # Print loss for each epoch
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} - Generator Loss (MSE): {g_loss.item()}, Discriminator Loss: {d_loss.item()}"
            )

    print("Training complete.")
