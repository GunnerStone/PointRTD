import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, token_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()

        # Transformer decoder with batch_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Linear layer to predict the denoised token for fake tokens
        self.token_predictor = nn.Linear(token_dim, token_dim)

    def forward(self, original_tokens, real_fake_labels):
        batch_size, seq_len, token_dim = original_tokens.shape
        generated_sequence = torch.zeros_like(original_tokens)

        for t in range(seq_len):
            real_mask = real_fake_labels[:, t].unsqueeze(-1).expand(-1, token_dim)
            new_generated_sequence = torch.where(
                real_mask,
                original_tokens[:, t, :],  # Use original token if uncorrupted
                generated_sequence[:, t, :].clone(),  # Otherwise, clean token
            )
            generated_sequence[:, t, :] = new_generated_sequence

            # Process only the fake tokens for cleaning
            fake_mask = ~real_fake_labels[:, t]  # Tokens marked as fake
            if fake_mask.any():  # If there are any fake tokens
                fake_indices = fake_mask.nonzero(as_tuple=True)[
                    0
                ]  # Batch indices of fake tokens
                prev_tokens = generated_sequence[:, : t + 1, :].clone()
                decoded_output = self.transformer_decoder(prev_tokens, original_tokens)
                current_token = decoded_output[:, -1, :]
                denoised_token = self.token_predictor(current_token)

                # Update only the fake tokens in the batch
                generated_sequence[fake_indices, t, :] = denoised_token[fake_indices, :]

        return generated_sequence


if __name__ == "__main__":
    # Example settings
    batch_size = 4
    seq_len = 10  # Number of tokens per sequence
    token_dim = 128  # Dimensionality of each token (embedding size)
    hidden_dim = 256  # Hidden layer dimension
    num_layers = 2  # Number of transformer decoder layers

    # Initialize the Generator
    generator = Generator(
        token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers
    )

    # Create a batch of example input tokens
    original_tokens = torch.randn(
        batch_size, seq_len, token_dim
    )  # Shape: (batch_size, seq_len, token_dim)

    # Example real/fake labels: True for real tokens, False for fake ones
    real_fake_labels = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    real_fake_labels[:, : seq_len // 2] = (
        True  # First half of the sequence is real, second half is fake
    )

    print(f"original_tokens shape: {original_tokens.shape}")
    print(f"real_fake_labels shape: {real_fake_labels.shape}")

    # Forward pass through the Generator
    generated_tokens = generator(original_tokens, real_fake_labels)

    # Output shape: (batch_size, seq_len, token_dim)
    print("Generated tokens shape:", generated_tokens.shape)

    # Sanity check: compare generated tokens with original tokens
    for i in range(batch_size):
        for t in range(seq_len):
            if real_fake_labels[i, t]:  # Real tokens should match the original tokens
                if torch.allclose(
                    generated_tokens[i, t], original_tokens[i, t], atol=1e-6
                ):
                    print(
                        f"Token {t} in sequence {i} is real and matches the original token."
                    )
                else:
                    print(
                        f"ERROR: Token {t} in sequence {i} is real but does not match the original token."
                    )
            else:  # Fake tokens should not match the original tokens
                if not torch.allclose(
                    generated_tokens[i, t], original_tokens[i, t], atol=1e-6
                ):
                    print(
                        f"Token {t} in sequence {i} is fake and was successfully denoised."
                    )
                else:
                    print(
                        f"ERROR: Token {t} in sequence {i} is fake but matches the original token."
                    )
