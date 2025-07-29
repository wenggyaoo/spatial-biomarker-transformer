import torch
import torch.nn as nn
import math


class SpatialPositionalEmbedding(nn.Module):
    """
    Create positional embeddings from 2D spatial coordinates
    """

    def __init__(self, d_model: int, max_distance: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.max_distance = max_distance

        # Create learnable position embedding
        self.x_embedding = nn.Linear(1, d_model // 2)
        self.y_embedding = nn.Linear(1, d_model // 2)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate positional embeddings from coordinates

        Args:
            coordinates: Tensor of shape (batch_size, seq_len, 2) with (x, y) coordinates

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = coordinates.shape

        # Normalize coordinates
        x_coords = coordinates[:, :, 0:1] / self.max_distance  # Shape: (batch_size, seq_len, 1)
        y_coords = coordinates[:, :, 1:2] / self.max_distance  # Shape: (batch_size, seq_len, 1)

        # Generate embeddings
        x_embed = self.x_embedding(x_coords)  # Shape: (batch_size, seq_len, d_model//2)
        y_embed = self.y_embedding(y_coords)  # Shape: (batch_size, seq_len, d_model//2)

        # Concatenate x and y embeddings
        pos_embed = torch.cat([x_embed, y_embed], dim=-1)  # Shape: (batch_size, seq_len, d_model)

        return pos_embed


class SinusoidalSpatialEmbedding(nn.Module):
    """
    Alternative sinusoidal positional embedding for 2D coordinates
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings from coordinates

        Args:
            coordinates: Tensor of shape (batch_size, seq_len, 2)

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = coordinates.shape
        device = coordinates.device

        # Create position encoding
        pos_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Generate frequency bands
        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2, device=device) *
                             -(math.log(10000.0) / (self.d_model // 2)))

        # X coordinate encoding
        x_coords = coordinates[:, :, 0].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        pos_embed[:, :, 0::4] = torch.sin(x_coords * div_term[::2])
        pos_embed[:, :, 1::4] = torch.cos(x_coords * div_term[::2])

        # Y coordinate encoding
        y_coords = coordinates[:, :, 1].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        pos_embed[:, :, 2::4] = torch.sin(y_coords * div_term[::2])
        pos_embed[:, :, 3::4] = torch.cos(y_coords * div_term[::2])

        return pos_embed