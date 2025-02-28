"""
Enhanced transformer implementations for soccer prediction.

This module provides advanced transformer architectures specifically
designed for soccer match prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedTabTransformer(nn.Module):
    """Enhanced TabTransformer with additional complexity and architectural improvements."""

    def __init__(
        self,
        input_dim,
        num_classes=None,
        is_regression=False,
        dim=128,
        depth=3,
        heads=8,
        dropout=0.2,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_multiplier=2.0,
        use_residual=True,
        use_layer_norm=True,
        gating_mechanism="glu",
        feature_interaction="cross",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.is_regression = is_regression

        # Input embedding
        self.embedding = nn.Linear(input_dim, dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Feature interaction module
        if feature_interaction == "cross":
            self.feature_interaction = CrossNetworkLayer(dim, num_layers=3)
        elif feature_interaction == "self":
            self.feature_interaction = SelfAttentionPooling(dim, heads)
        else:
            self.feature_interaction = nn.Identity()

        # Positional embedding - fix dimension handling
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(depth):
            self.transformer_layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim) if use_layer_norm else nn.Identity(),
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            batch_first=True,
                        ),
                        nn.LayerNorm(dim) if use_layer_norm else nn.Identity(),
                        FeedForward(
                            dim=dim,
                            hidden_dim=int(dim * mlp_hidden_multiplier),
                            dropout=ff_dropout,
                            activation="gelu",
                        ),
                    ]
                )
            )

        # Gating mechanism
        if gating_mechanism == "glu":
            self.gate = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GLU())
        elif gating_mechanism == "selu":
            self.gate = nn.SELU()
        else:
            self.gate = nn.Identity()

        # Final layers
        self.layer_norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()

        # Output head
        if is_regression:
            self.out_dim = 1
            self.output_activation = nn.Identity()
        else:
            self.out_dim = num_classes
            self.output_activation = nn.LogSoftmax(dim=1)

        # Output layers with residual blocks
        self.out_layers = nn.Sequential(
            ResidualBlock(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, self.out_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Input embedding
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        # Add sequence dimension for attention
        x = x.unsqueeze(1)

        # Add positional embedding
        x = x + self.pos_embedding

        # Feature interaction - handle dimensions consistently
        if not isinstance(self.feature_interaction, nn.Identity):
            x_interaction = self.feature_interaction(x)
            x = x + x_interaction

        # Transformer layers
        for ln1, attn, ln2, ff in self.transformer_layers:
            # Self-attention block
            x_norm = ln1(x)
            x_attn = attn(x_norm, x_norm, x_norm)[0]
            x = x + x_attn  # Residual connection

            # Feed-forward block
            x_norm = ln2(x)
            x_ff = ff(x_norm)
            x = x + x_ff  # Residual connection

        # Remove sequence dimension and apply gating
        x = x.squeeze(1)
        x = self.gate(x)

        # Final layer norm
        x = self.layer_norm(x)

        # Output projection
        x = self.out_layers(x)

        # Apply output activation
        x = self.output_activation(x)

        return x


class FeedForward(nn.Module):
    """
    Feed-forward network with residual connection.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0, activation="relu"):
        super().__init__()

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "selu":
            act_fn = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(self.dim)
        self.ff = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim)
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class CrossNetworkLayer(nn.Module):
    """
    Cross Network for feature interactions.
    """

    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # Weight matrices
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(dim, dim) * 0.02) for _ in range(num_layers)]
        )

        # Bias terms
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(num_layers)]
        )

    def forward(self, x):
        # Preserve batch dimension
        # Input shape: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        x0 = x

        for i in range(self.num_layers):
            # Reshape for matrix multiplication while preserving batch dimension
            # (batch_size, seq_len, dim) -> (batch_size * seq_len, dim)
            x_flat = x.view(-1, dim)

            # Matrix multiplication for cross terms
            xT = torch.matmul(x_flat, self.weights[i])

            # Reshape back to (batch_size, seq_len, dim)
            xT = xT.view(batch_size, seq_len, dim)

            # Element-wise multiplication with original input + bias
            x = x0 * xT + self.bias[i].view(1, 1, dim) + x

        return x


class SelfAttentionPooling(nn.Module):
    """
    Self-attention pooling for feature interaction.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output


# Function to optimize hyperparameters with Optuna
def optimize_hyperparameters(X, y, is_classification, n_trials=100, cv=5):
    """
    Optimize hyperparameters using Optuna.

    Args:
        X: Training features
        y: Target values
        is_classification: Whether this is a classification task
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds

    Returns:
        Best hyperparameters
    """
    import optuna
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    import numpy as np
    import logging
    from functools import partial
    from sklearn.pipeline import Pipeline

    def objective(trial, X, y, is_classification, cv):
        # Define hyperparameters to search
        params = {
            "dim": trial.suggest_categorical("dim", [64, 128, 256, 512]),
            "depth": trial.suggest_int("depth", 2, 6),
            "heads": trial.suggest_categorical("heads", [4, 8, 12, 16]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.1, 0.5),
            "ff_dropout": trial.suggest_float("ff_dropout", 0.1, 0.5),
            "mlp_hidden_multiplier": trial.suggest_float(
                "mlp_hidden_multiplier", 1.0, 4.0
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            "use_layer_norm": trial.suggest_categorical(
                "use_layer_norm", [True, False]
            ),
        }

        # Import necessary components
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Create preprocessing pipeline
        # This is a simplified version - in practice, use create_preprocessing_pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        # Identify column types
        if hasattr(X, "select_dtypes"):
            numeric_cols = X.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Create preprocessors
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ]
            )
        else:
            # If X is not a DataFrame, just use a standard scaler
            preprocessor = StandardScaler()

        # Select model
        from src.ai_models.enhanced_transformer import EnhancedTabTransformer
        from src.ai_models.model_wrapper import EnhancedPyTorchModelWrapper

        # Create model wrapper
        model = EnhancedPyTorchModelWrapper(
            model_class=EnhancedTabTransformer,
            model_params={
                "dim": params["dim"],
                "depth": params["depth"],
                "heads": params["heads"],
                "dropout": params["dropout"],
                "attn_dropout": params["attn_dropout"],
                "ff_dropout": params["ff_dropout"],
                "mlp_hidden_multiplier": params["mlp_hidden_multiplier"],
                "use_residual": params["use_residual"],
                "use_layer_norm": params["use_layer_norm"],
            },
            is_regression=not is_classification,
            batch_size=params["batch_size"],
            num_epochs=50,  # Reduced for faster trials
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            early_stopping_patience=5,
            verbose=False,  # Reduce output during optimization
        )

        # Create pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Cross-validation
        if is_classification:
            # Use stratified CV for classification
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = "f1_weighted"
        else:
            # Use regular CV for regression
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = "neg_mean_squared_error"

        try:
            scores = cross_val_score(pipeline, X, y, cv=cv_obj, scoring=scoring)

            if is_classification:
                # For classification, higher is better
                return np.mean(scores)
            else:
                # For regression, RMSE lower is better
                return -np.sqrt(-np.mean(scores))
        except Exception as e:
            logging.warning(f"Trial failed with error: {str(e)}")
            # Return a bad score to discourage this combination of parameters
            return -np.inf if is_classification else np.inf

    # Create study
    if is_classification:
        study = optuna.create_study(direction="maximize")
    else:
        study = optuna.create_study(direction="minimize")

    # Run optimization
    objective_func = partial(
        objective, X=X, y=y, is_classification=is_classification, cv=cv
    )
    study.optimize(objective_func, n_trials=n_trials)

    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_trial.value:.4f}")
    logging.info(f"Best parameters: {study.best_trial.params}")

    # Return best parameters
    return study.best_trial.params
