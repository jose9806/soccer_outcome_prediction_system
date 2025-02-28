"""
TabTransformer implementation for tabular data prediction.

This module provides a transformer-based architecture specifically designed
for tabular data prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabTransformer(nn.Module):
    """
    A transformer-based model adapted for tabular data.

    This architecture uses a feature tokenizer to convert tabular features into
    token embeddings, then applies transformer layers to capture feature interactions.
    """

    def __init__(
        self,
        input_dim,
        num_classes=None,
        dim=64,
        depth=3,
        heads=8,
        dim_head=16,
        dropout=0.2,
        is_regression=False,
    ):
        """Initialize the TabTransformer model.

        Args:
            input_dim: Number of input features
            num_classes: Number of output classes for classification
            dim: Embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            dropout: Dropout rate
            is_regression: Whether this is a regression task
        """
        super().__init__()

        self.is_regression = is_regression
        self.input_dim = input_dim
        self.dim = dim

        # Feature embedding layer
        self.embedding = nn.Linear(input_dim, dim)

        # Embedding normalization
        self.norm = nn.LayerNorm(dim)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        nn.MultiheadAttention(dim, heads, dropout=dropout),
                        nn.LayerNorm(dim),
                        nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 4, dim),
                            nn.Dropout(dropout),
                        ),
                    ]
                )
            )

        # Output layer
        self.out_norm = nn.LayerNorm(dim)
        if is_regression:
            self.fc_out = nn.Linear(dim, 1)
        else:
            self.fc_out = nn.Linear(dim, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, features)

        Returns:
            Output tensor of shape (batch_size, num_classes) or (batch_size, 1)
        """
        # Project input features to embedding space
        # (batch_size, features) -> (batch_size, dim)
        x = self.embedding(x)
        x = self.norm(x)

        # Create a "fake" sequence dimension for attention
        # (batch_size, dim) -> (1, batch_size, dim)
        x = x.unsqueeze(0)

        # Apply transformer layers
        for norm1, attn, norm2, ff in self.transformer_layers:
            # Self-attention block
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out

            # Feed-forward block
            x_norm = norm2(x)
            x = x + ff(x_norm)

        # Remove sequence dimension and normalize
        # (1, batch_size, dim) -> (batch_size, dim)
        x = x.squeeze(0)
        x = self.out_norm(x)

        # Final projection
        x = self.fc_out(x)

        # Apply log_softmax for classification
        if not self.is_regression and x.shape[-1] > 1:
            x = F.log_softmax(x, dim=-1)

        return x


class TabTransformerWithFeatureTokenizer(nn.Module):
    """
    Advanced TabTransformer with feature-wise tokenization.

    This version treats each feature as a separate token, allowing the transformer
    to model interactions between individual features directly.
    """

    def __init__(
        self,
        input_dim,
        num_classes=None,
        dim=64,
        depth=3,
        heads=8,
        dropout=0.2,
        is_regression=False,
    ):
        """Initialize the TabTransformer with feature tokenizer.

        Args:
            input_dim: Number of input features
            num_classes: Number of output classes for classification
            dim: Embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dropout: Dropout rate
            is_regression: Whether this is a regression task
        """
        super().__init__()

        self.is_regression = is_regression
        self.input_dim = input_dim

        # Feature embeddings - one per input feature
        self.feature_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)
                )
                for _ in range(input_dim)
            ]
        )

        # Positional embeddings to distinguish features
        self.pos_embedding = nn.Parameter(torch.randn(input_dim, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # Output layer
        if is_regression:
            self.fc_out = nn.Linear(dim, 1)
        else:
            self.fc_out = nn.Linear(dim, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, features)

        Returns:
            Output tensor of shape (batch_size, num_classes) or (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Process each feature independently
        embedded_features = []
        for i in range(self.input_dim):
            # Extract the i-th feature and reshape
            # (batch_size,) -> (batch_size, 1)
            feature = x[:, i : i + 1]

            # Embed the feature
            # (batch_size, 1) -> (batch_size, dim)
            embedded = self.feature_embeddings[i](feature)
            embedded_features.append(embedded)

        # Stack embedded features
        # List of (batch_size, dim) -> (batch_size, input_dim, dim)
        x = torch.stack(embedded_features, dim=1)

        # Add positional embeddings
        # (1, input_dim, dim) + (batch_size, input_dim, dim)
        x = x + self.pos_embedding.unsqueeze(0)

        # Transpose for transformer: (batch_size, input_dim, dim) -> (input_dim, batch_size, dim)
        x = x.transpose(0, 1)

        # Apply transformer
        x = self.transformer_encoder(x)

        # Average pooling across features
        # (input_dim, batch_size, dim) -> (batch_size, dim)
        x = x.transpose(0, 1).mean(dim=1)

        # Output projection
        x = self.fc_out(x)

        # Apply log_softmax for classification
        if not self.is_regression and x.shape[-1] > 1:
            x = F.log_softmax(x, dim=-1)

        return x
