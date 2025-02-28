"""
Enhanced PyTorch model wrapper with advanced training capabilities.

This module provides an enhanced wrapper for PyTorch models compatible
with scikit-learn API and offering advanced training features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



class EnhancedPyTorchModelWrapper:
    """
    Enhanced wrapper for PyTorch models with additional training features.
    """

    def __init__(
        self,
        model_class,
        model_params,
        is_regression=False,
        batch_size=64,
        num_epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_patience=10,
        lr_scheduler_type="plateau",  # "plateau", "onecycle", "cosine"
        mixup_alpha=0.0,  # If > 0, use mixup augmentation
        focal_loss_gamma=0.0,  # If > 0, use focal loss
        label_smoothing=0.0,  # Label smoothing factor
        gradient_clip_val=0.0,  # Gradient clipping
        verbose=True,
        class_weights=None,  # Class weights for imbalanced data
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.is_regression = is_regression
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.mixup_alpha = mixup_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.label_smoothing = label_smoothing
        self.gradient_clip_val = gradient_clip_val
        self.verbose = verbose
        self.class_weights = class_weights

        self.model = None

        if self.verbose:
            logging.info(f"Using device: {self.device}")

    def fit(self, X, y):
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Initialize model
        input_dim = X.shape[1]
        if self.is_regression:
            num_classes = None
            y = y.astype(np.float32)
        else:
            # For classification, set the classes_ attribute needed by sklearn
            unique_classes = np.unique(y)
            self.classes_ = unique_classes
            num_classes = len(unique_classes)
            y = y.astype(np.int64)

            # Compute class weights if not provided
            if self.class_weights is None and num_classes > 0:
                class_counts = np.bincount(y)
                self.class_weights = 1.0 / class_counts
                self.class_weights = (
                    self.class_weights
                    / np.sum(self.class_weights)
                    * len(self.class_weights)
                )
                self.class_weights = torch.FloatTensor(self.class_weights).to(
                    self.device
                )

        # Create model
        self.model = self.model_class(
            input_dim=input_dim,
            num_classes=num_classes,
            is_regression=self.is_regression,
            **self.model_params,
        ).to(self.device)

        # Prepare data loaders
        X_tensor = torch.FloatTensor(X).to(self.device)
        if self.is_regression:
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        else:
            y_tensor = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Loss function
        if self.is_regression:
            criterion = nn.MSELoss()
        else:
            if self.focal_loss_gamma > 0:
                criterion = FocalLoss(
                    alpha=self.class_weights,
                    gamma=self.focal_loss_gamma,
                    reduction="mean",
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=self.class_weights, label_smoothing=self.label_smoothing
                )

        # Learning rate scheduler
        if self.lr_scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=self.verbose
            )
        elif self.lr_scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.num_epochs,
                steps_per_epoch=len(dataloader),
            )
        elif self.lr_scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )

        # Training loop
        self.model.train()
        best_loss = float("inf")
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Apply mixup augmentation if enabled
                if not self.is_regression and self.mixup_alpha > 0:
                    batch_X, batch_y_a, batch_y_b, lam = self._mixup_data(
                        batch_X, batch_y, self.mixup_alpha
                    )

                # Forward pass
                outputs = self.model(batch_X)

                # Calculate loss with mixup if enabled
                if not self.is_regression and self.mixup_alpha > 0:
                    loss = self._mixup_criterion(
                        criterion, outputs, batch_y_a, batch_y_b, lam
                    )
                else:
                    if self.is_regression:
                        loss = criterion(outputs, batch_y)
                    else:
                        if batch_y.dim() > 1 and batch_y.size(1) == 1:
                            batch_y = batch_y.squeeze(1)
                        loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()

                # Apply gradient clipping if enabled
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                optimizer.step()

                # Update learning rate for batch-based schedulers
                if self.lr_scheduler_type == "onecycle":
                    scheduler.step()

                total_loss += loss.item()

            # Calculate average loss
            avg_loss = total_loss / len(dataloader)

            # Update learning rate for epoch-based schedulers
            if self.lr_scheduler_type == "plateau":
                scheduler.step(avg_loss)
            elif self.lr_scheduler_type == "cosine":
                scheduler.step()

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict().copy()
            else:
                epochs_without_improvement += 1

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def _mixup_data(self, x, y, alpha=1.0):
        """
        Apply mixup augmentation.

        Args:
            x: Input features
            y: Target values
            alpha: Mixup alpha parameter

        Returns:
            Mixed inputs, targets a, targets b, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Apply mixup criterion.

        Args:
            criterion: Loss function
            pred: Predictions
            y_a: Targets a
            y_b: Targets b
            lam: Lambda value

        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

            if self.is_regression:
                return predictions.cpu().numpy().flatten()
            else:
                return predictions.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if self.is_regression:
            raise ValueError("predict_proba() not supported for regression tasks.")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            proba = torch.exp(predictions)

            # Ensure probabilities match the order of classes_
            # This is important for scikit-learn compatibility
            return proba.cpu().numpy()


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for imbalanced classification.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction="none", weight=self.alpha)(
            inputs, targets
        )
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class EnsembleTransformerModel:
    """
    Ensemble of transformer models for improved prediction accuracy.
    """

    def __init__(
        self,
        is_classification=True,
        ensemble_type="stacking",  # 'voting' or 'stacking'
        n_models=3,
        model_types=["basic", "feature_tokenizer", "enhanced"],
        use_weights=True,
        meta_learner=None,
        meta_learner_params=None,
        cv=5,
        hp_tuning=True,
        n_trials=50,
        random_state=42,
    ):
        self.is_classification = is_classification
        self.ensemble_type = ensemble_type
        self.n_models = n_models
        self.model_types = model_types
        self.use_weights = use_weights
        self.meta_learner = meta_learner
        self.meta_learner_params = meta_learner_params or {}
        self.cv = cv
        self.hp_tuning = hp_tuning
        self.n_trials = n_trials
        self.random_state = random_state
        self.ensemble = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Import necessary components
        from src.ai_models.tab_transformer import (
            TabTransformer,
            TabTransformerWithFeatureTokenizer,
        )
        from src.ai_models.enhanced_transformer import EnhancedTabTransformer

        # Log important information for debugging
        logging.info(
            f"Creating {self.ensemble_type} ensemble for {'classification' if self.is_classification else 'regression'} task"
        )

        # Prepare models for the ensemble
        models = []

        # Initialize preprocessor once to ensure consistency
        preprocessor = self._create_preprocessor(X_train)

        # Fixed dimension for all models to ensure consistency
        fixed_dim = 128

        for i in range(self.n_models):
            # Select model type (cycle through if needed)
            model_type = self.model_types[i % len(self.model_types)]

            # Create a unique model name
            model_name = f"transformer_{model_type}_{i}"

            # Base parameters that work for all models
            base_params = {
                "dim": fixed_dim,
                "depth": 3 + i % 3,
                "heads": 8,
                "dropout": 0.2 + (i * 0.05) % 0.2,
            }

            # Select model class and adjust parameters accordingly
            if model_type == "basic":
                model_class = TabTransformer
                model_params = base_params.copy()  # Only use base parameters
            elif model_type == "feature_tokenizer":
                model_class = TabTransformerWithFeatureTokenizer
                model_params = base_params.copy()  # Only use base parameters
            else:  # "enhanced"
                model_class = EnhancedTabTransformer
                # Add enhanced parameters for this model type
                model_params = base_params.copy()
                model_params.update(
                    {
                        "attn_dropout": 0.1 + (i * 0.05) % 0.2,
                        "ff_dropout": 0.1 + (i * 0.05) % 0.2,
                        "mlp_hidden_multiplier": 2.0,
                        "use_residual": True,
                        "use_layer_norm": True,
                        "gating_mechanism": "glu",
                        "feature_interaction": "cross",
                    }
                )

            # Create model wrapper with appropriate parameters
            model = EnhancedPyTorchModelWrapper(
                model_class=model_class,
                model_params=model_params,  # Use filtered parameters
                is_regression=not self.is_classification,
                batch_size=64 + (i * 32) % 128,  # Vary batch size
                num_epochs=150 if model_type == "enhanced" else 100,
                learning_rate=1e-3 * (0.8 + 0.4 * (i % 3)),
                weight_decay=1e-4 * (0.5 + i % 3),
                early_stopping_patience=15 if model_type == "enhanced" else 10,
                lr_scheduler_type=["plateau", "cosine", "onecycle"][i % 3],
                mixup_alpha=0.2 if self.is_classification else 0.0,
                focal_loss_gamma=2.0 if self.is_classification else 0.0,
                gradient_clip_val=1.0 if model_type == "enhanced" else 0.0,
                verbose=True,
            )

            # Create pipeline
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            # Add to models list
            models.append((model_name, pipeline))

        # Create appropriate ensemble
        if self.ensemble_type == "voting":
            self._create_voting_ensemble(models)
        else:
            self._create_stacking_ensemble(models)

        # Train the ensemble
        logging.info(
            f"Training {self.ensemble_type} ensemble with {self.n_models} models"
        )
        self.ensemble.fit(X_train, y_train)

        return self

    def _create_preprocessor(self, X_train):
        """Create appropriate preprocessor for the data."""
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        # Identify column types if X_train is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            numeric_cols = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_cols = X_train.select_dtypes(
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
            return ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ]
            )
        else:
            # If X_train is not a DataFrame, just use a standard scaler
            return StandardScaler()

    def _create_voting_ensemble(self, models):
        """Create voting ensemble appropriate for the task type."""
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        if self.is_classification:
            self.ensemble = VotingClassifier(
                estimators=models,
                voting="soft",
                weights=[1] * self.n_models if not self.use_weights else None,
                n_jobs=1,  # PyTorch models don't work well with parallel processing
            )
        else:
            self.ensemble = VotingRegressor(
                estimators=models,
                weights=[1] * self.n_models if not self.use_weights else None,
                n_jobs=1,
            )

    def _create_stacking_ensemble(self, models):
        """Create stacking ensemble appropriate for the task type."""
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import StratifiedKFold, KFold

        # Select meta-learner
        if self.meta_learner is None:
            if self.is_classification:
                meta = LogisticRegression(max_iter=1000, **self.meta_learner_params)
            else:
                meta = Ridge(**self.meta_learner_params)
        else:
            meta = self.meta_learner(**self.meta_learner_params)

        # Configure CV
        if self.is_classification:
            cv_obj = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        else:
            cv_obj = KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )

        # Create stacking ensemble
        if self.is_classification:
            self.ensemble = StackingClassifier(
                estimators=models,
                final_estimator=meta,
                cv=cv_obj,
                stack_method="predict_proba",
                n_jobs=1,
            )
        else:
            self.ensemble = StackingRegressor(
                estimators=models, final_estimator=meta, cv=cv_obj, n_jobs=1
            )

    def predict(self, X):
        """Make predictions with the ensemble."""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities with the ensemble."""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call fit() first.")

        if not self.is_classification:
            raise ValueError("predict_proba() not supported for regression tasks.")

        return self.ensemble.predict_proba(X)


# Feature engineering components
class TimeFeatureGenerator:
    """
    Generate advanced time-based features from date columns.
    """

    def __init__(self, date_col="match_date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.date_col not in X.columns:
            return X

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
            X[self.date_col] = pd.to_datetime(X[self.date_col])

        # Extract date components
        X["day_of_week"] = X[self.date_col].dt.dayofweek
        X["month"] = X[self.date_col].dt.month
        X["day"] = X[self.date_col].dt.day
        X["hour"] = X[self.date_col].dt.hour
        X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)

        # Season phase (early, mid, late)
        # Assuming season starts in August (month 8)
        X["month_of_season"] = (X["month"] - 8) % 12
        X["season_phase"] = pd.cut(
            X["month_of_season"],
            bins=[-1, 2, 6, 11],
            labels=[0, 1, 2],  # Early: 0, Mid: 1, Late: 2
        ).astype(int)

        # Time-based features specific to soccer
        # Is it an important time of the season?
        X["is_season_end"] = ((X["month"] >= 4) & (X["month"] <= 5)).astype(int)

        # Drop original date column
        X = X.drop(columns=[self.date_col])

        return X


def create_advanced_preprocessing_pipeline(X_train, use_time_features=True):
    """
    Create an advanced preprocessing pipeline with enhanced feature engineering.

    Args:
        X_train: Training features
        use_time_features: Whether to include time features

    Returns:
        Pipeline: Preprocessing pipeline
    """
    # Identify column types
    if isinstance(X_train, pd.DataFrame):
        numeric_cols = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove special columns
        date_col = "match_date" if "match_date" in X_train.columns else None
        special_cols = [col for col in [date_col] if col]
        numeric_cols = [col for col in numeric_cols if col not in special_cols]
        categorical_cols = [col for col in categorical_cols if col not in special_cols]

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

        # Create transformers list
        transformers = [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]

        # Add time feature generator if enabled
        if use_time_features and date_col:
            time_transformer = Pipeline(
                [("time_features", TimeFeatureGenerator(date_col=date_col))]
            )

            # Add to transformers
            transformers.append(("time", time_transformer, [date_col]))

        # Create preprocessor
        preprocessor = ColumnTransformer(transformers=transformers)
    else:
        # If X_train is not a DataFrame, just use a standard scaler
        preprocessor = StandardScaler()

    return preprocessor
