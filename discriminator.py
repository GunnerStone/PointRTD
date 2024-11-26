import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, token_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()

        # Transformer encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head for binary predictions
        self.classifier = nn.Linear(token_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Transformer encoder with batch_first=True
        encoded_tokens = self.transformer_encoder(
            x
        )  # Shape: (batch_size, seq_len, token_dim)
        logits = self.classifier(encoded_tokens)  # Shape: (batch_size, seq_len, 1)
        probs = self.sigmoid(logits)  # Binary classification for each token

        return probs


if __name__ == "__main__":
    # Define the binary cross-entropy loss
    bce_loss = nn.BCELoss()

    # Example settings (from before)
    batch_size = 4
    seq_len = 10
    token_dim = 128

    # Initialize the Discriminator (from the previous code)
    discriminator = Discriminator(token_dim=token_dim, hidden_dim=256, num_layers=2)

    # Example input: A batch of token sequences
    example_input = torch.randn(batch_size, seq_len, token_dim)

    # Forward pass through the Discriminator to get the output probabilities
    output = discriminator(example_input)

    print("Output shape:", output.shape)

    # Define the target labels for each token in the sequence (same shape as the output)
    # Let's assume for this example:
    # - Half of the tokens are real (labeled as 1) and the other half are fake (labeled as 0)
    targets = torch.zeros_like(output)  # Start with all fake (0)
    targets[:, : seq_len // 2, :] = 1  # Mark the first half of tokens as real (1)

    # Calculate the binary cross-entropy loss
    loss = bce_loss(output, targets)

    print("Targets shape:", targets.shape)
    print("Loss:", loss.item())
