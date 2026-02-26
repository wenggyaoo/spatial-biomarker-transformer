import torch
import torch.nn as nn
import math

class SinusoidalSpatialEmbedding(nn.Module):
    """
    Creates fixed sinusoidal positional embeddings from 2D Cartesian coordinates (x, y).
    This is a 2D adaptation of the standard positional encoding from "Attention Is All You Need".
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 to evenly split between x/y sin/cos, but got {d_model}")
        self.d_model = d_model

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings from coordinates.

        Args:
            coordinates: Tensor of shape (batch_size, seq_len, 2) with (x, y) coordinates.

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = coordinates.shape
        device = coordinates.device

        # Create position encoding matrix
        pos_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Generate frequency bands for a quarter of the model dimension, as we will apply
        # sin/cos to both x and y, effectively using 4 parts.
        half_d_model = self.d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, 2, device=device) *
                             -(math.log(10000.0) / half_d_model))

        x_coords = coordinates[:, :, 0].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        y_coords = coordinates[:, :, 1].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

        # Interleave the embeddings for x and y coordinates
        # Example for d_model=512:
        # x -> channels 0,1,4,5,8,9,...
        # y -> channels 2,3,6,7,10,11,...
        pos_embed[:, :, 0::4] = torch.sin(x_coords * div_term)
        pos_embed[:, :, 1::4] = torch.cos(x_coords * div_term)
        pos_embed[:, :, 2::4] = torch.sin(y_coords * div_term)
        pos_embed[:, :, 3::4] = torch.cos(y_coords * div_term)

        return pos_embed


class SinusoidalRotationalEmbedding(nn.Module):
    """
    Creates fixed sinusoidal positional embeddings from 2D polar coordinates (radius and angle).
    This embedding is inherently more robust to rotations of the input data.
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even to split between radius and angle, but got {d_model}")
        self.d_model = d_model

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings from relative Cartesian coordinates.
        The input coordinates are expected to be relative to a center point (0,0).

        Args:
            coordinates: Tensor of shape (batch_size, seq_len, 2) with relative (x, y) coordinates.

        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = coordinates.shape
        device = coordinates.device

        # --- Convert Cartesian to Polar Coordinates ---
        x = coordinates[:, :, 0]
        y = coordinates[:, :, 1]
        # Radius 'r' is the distance from the origin (center cell)
        radius = torch.sqrt(x**2 + y**2).unsqueeze(-1) # Shape: (batch_size, seq_len, 1)
        # Angle 'theta' is the angle relative to the positive x-axis
        angle = torch.atan2(y, x).unsqueeze(-1)      # Shape: (batch_size, seq_len, 1)

        # --- Create Sinusoidal Embeddings from Polar Coordinates ---
        pos_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # We dedicate half of the embedding dimension to radius and half to angle.
        half_d_model = self.d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, 2, device=device) *
                             -(math.log(10000.0) / half_d_model))

        # Encode Radius in the first half of the embedding channels
        pos_embed[:, :, 0:half_d_model:2] = torch.sin(radius * div_term)
        pos_embed[:, :, 1:half_d_model:2] = torch.cos(radius * div_term)

        # Encode Angle in the second half of the embedding channels
        pos_embed[:, :, half_d_model::2] = torch.sin(angle * div_term)
        pos_embed[:, :, half_d_model + 1::2] = torch.cos(angle * div_term)

        return pos_embed