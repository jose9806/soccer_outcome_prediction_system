"""
Train enhanced transformer-based prediction models for soccer.

This script trains advanced transformer models for soccer match prediction tasks
with hyperparameter tuning, ensemble modeling, and improved feature engineering.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import logging
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original models
from src.ai_models.tab_transformer import (
    TabTransformer,
    TabTransformerWithFeatureTokenizer,
)

# Import enhanced models and components
from src.ai_models.enhanced_transformer import (
    EnhancedTabTransformer,
    optimize_hyperparameters,
)
from src.ai_models.model_wrapper import (
    EnhancedPyTorchModelWrapper,
    EnsembleTransformerModel,
    create_advanced_preprocessing_pipeline,
)

# Original imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


def prepare_data(df, target_col, test_size=0.2, time_based=True):
    """Prepare data for training and testing.

    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Test set proportion
        time_based: Whether to use time-based split

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop non-feature columns
    non_feature_cols = [
        "match_id",
        "match_date",
        "date",
        "home_team",
        "away_team",
        "season",
        "stage",
    ]

    # Store important columns for later use
    important_cols = {}
    for col in non_feature_cols:
        if col in X.columns:
            important_cols[col] = X[col]

    X = X.drop(columns=[col for col in non_feature_cols if col in X.columns])

    # Use time-based split if possible
    if time_based and "match_date" in df.columns:
        # Sort by date
        df = df.sort_values("match_date")

        # Split by time
        train_size = int(len(df) * (1 - test_size))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(
            columns=[target_col]
            + [col for col in non_feature_cols if col in train_df.columns]
        )
        y_train = train_df[target_col]

        X_test = test_df.drop(
            columns=[target_col]
            + [col for col in non_feature_cols if col in test_df.columns]
        )
        y_test = test_df[target_col]
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    return X_train, X_test, y_train, y_test, important_cols


def create_preprocessing_pipeline(X_train):
    """Create a preprocessing pipeline.

    Args:
        X_train: Training features

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    # Identify column types
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Remove 'team' columns from categorical for special handling
    team_cols = [col for col in categorical_cols if "team" in col.lower()]
    categorical_cols = [col for col in categorical_cols if col not in team_cols]

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

    team_transformer = Pipeline(
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
            ("team", team_transformer, team_cols),
        ]
    )

    return preprocessor


def train_transformer_model(
    X_train,
    X_test,
    y_train,
    y_test,
    is_classification,
    model_type="basic",
    use_ensemble=False,
    ensemble_type="stacking",
    n_models=3,
    perform_hp_tuning=False,
    n_trials=50,
    use_advanced_features=False,
):
    """Train a transformer model with enhanced capabilities.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        is_classification: Whether this is a classification task
        model_type: Type of transformer model ('basic', 'feature_tokenizer', or 'enhanced')
        use_ensemble: Whether to use ensemble modeling
        ensemble_type: Type of ensemble ('voting' or 'stacking')
        n_models: Number of models in the ensemble
        perform_hp_tuning: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter tuning trials
        use_advanced_features: Whether to use advanced feature engineering

    Returns:
        Trained model pipeline and evaluation metrics
    """
    # Create preprocessing pipeline
    if use_advanced_features:
        preprocessor = create_advanced_preprocessing_pipeline(
            X_train, use_time_features=True
        )
    else:
        preprocessor = create_preprocessing_pipeline(X_train)

    if use_ensemble:
        # Create and train ensemble model
        logging.info(f"Training {ensemble_type} ensemble with {n_models} models")

        ensemble = EnsembleTransformerModel(
            is_classification=is_classification,
            ensemble_type=ensemble_type,
            n_models=n_models,
            model_types=["basic", "feature_tokenizer", "enhanced"],
            hp_tuning=perform_hp_tuning,
            n_trials=n_trials,
        )

        # Create pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", ensemble)])

        # Train model
        pipeline.fit(X_train, y_train)

    else:
        # Select model class
        if model_type == "basic":
            model_class = TabTransformer
        elif model_type == "feature_tokenizer":
            model_class = TabTransformerWithFeatureTokenizer
        else:  # "enhanced"
            model_class = EnhancedTabTransformer

        # Default hyperparameters
        model_params = {"dim": 64, "depth": 3, "heads": 8, "dropout": 0.2}

        # Perform hyperparameter tuning if requested
        if perform_hp_tuning:
            logging.info("Starting hyperparameter tuning")
            best_params = optimize_hyperparameters(
                X_train, y_train, is_classification, n_trials=n_trials
            )
            # Update model parameters with best parameters
            model_params.update(best_params)
            logging.info(f"Best hyperparameters: {best_params}")

        # Create model wrapper
        if model_type == "enhanced":
            # Use enhanced wrapper for enhanced model
            model = EnhancedPyTorchModelWrapper(
                model_class=model_class,
                model_params=model_params,
                is_regression=not is_classification,
                batch_size=model_params.get("batch_size", 64),
                num_epochs=150,
                learning_rate=model_params.get("learning_rate", 1e-3),
                weight_decay=model_params.get("weight_decay", 1e-4),
                early_stopping_patience=15,
                lr_scheduler_type="cosine",
                mixup_alpha=0.2 if is_classification else 0.0,
                focal_loss_gamma=2.0 if is_classification else 0.0,
                gradient_clip_val=1.0,
                verbose=True,
            )
        else:
            # Use original wrapper for basic models
            from scripts.train_transformer_models import PyTorchModelWrapper

            model = PyTorchModelWrapper(
                model_class=model_class,
                model_params=model_params,
                is_regression=not is_classification,
                batch_size=model_params.get("batch_size", 64),
                num_epochs=100,
                learning_rate=model_params.get("learning_rate", 1e-3),
                early_stopping_patience=10,
                verbose=True,
            )

        # Create pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Train model
        logging.info(f"Training {model_type} transformer model...")
        pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        if len(np.unique(y_train)) > 2:  # Multi-class
            f1 = f1_score(y_test, y_pred, average="weighted")
        else:  # Binary
            f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": report,
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        metrics = {"mse": mse, "rmse": rmse, "mae": mae}

    return pipeline, metrics


def save_model(model, metrics, filepath):
    """Save model and metrics.

    Args:
        model: Trained model
        metrics: Evaluation metrics
        filepath: Path to save model
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model
    joblib.dump(model, filepath)

    # Save metrics
    metrics_path = f"{os.path.splitext(filepath)[0]}_metrics.json"

    # Convert metrics to serializable format
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = str(value)

    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def train_match_outcome_models(data_path, output_path, model_config):
    """Train match outcome prediction models with transformers.

    Args:
        data_path: Path to datasets
        output_path: Path to save models
        model_config: Model configuration dictionary
    """
    # Create output directories
    os.makedirs(os.path.join(output_path, "outcome"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "expected_goals"), exist_ok=True)

    # Get model configuration
    model_type = model_config.get("model_type", "basic")
    use_ensemble = model_config.get("use_ensemble", False)
    ensemble_type = model_config.get("ensemble_type", "stacking")
    n_models = model_config.get("n_models", 3)
    perform_hp_tuning = model_config.get("perform_hp_tuning", False)
    n_trials = model_config.get("n_trials", 50)
    use_advanced_features = model_config.get("use_advanced_features", False)

    # Train 3-way outcome prediction model
    outcome_path = os.path.join(data_path, "match_outcome", "match_outcome_3way.csv")
    if os.path.exists(outcome_path):
        logging.info(f"Training 3-way outcome prediction model from {outcome_path}")
        outcome_df = pd.read_csv(outcome_path)

        # Convert date column if present
        if "match_date" in outcome_df.columns:
            outcome_df["match_date"] = pd.to_datetime(outcome_df["match_date"])

        # Prepare data
        X_train, X_test, y_train, y_test, _ = prepare_data(outcome_df, "outcome_num")

        # Train model
        model, metrics = train_transformer_model(
            X_train,
            X_test,
            y_train,
            y_test,
            is_classification=True,
            model_type=model_type,
            use_ensemble=use_ensemble,
            ensemble_type=ensemble_type,
            n_models=n_models,
            perform_hp_tuning=perform_hp_tuning,
            n_trials=n_trials,
            use_advanced_features=use_advanced_features,
        )

        # Model name with configuration
        model_name = f"match_outcome_3way_transformer_{model_type}"
        if use_ensemble:
            model_name += f"_ensemble_{ensemble_type}"
        if perform_hp_tuning:
            model_name += "_tuned"
        if use_advanced_features:
            model_name += "_advanced"

        # Save model
        model_path = os.path.join(output_path, "outcome", f"{model_name}.joblib")
        save_model(model, metrics, model_path)

        logging.info(f"Saved 3-way outcome model to {model_path}")
        logging.info(
            f"Model metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
        )

    # Train expected goals prediction models
    for target in ["home_goals", "away_goals", "total_goals"]:
        target_path = os.path.join(data_path, "match_outcome", f"{target}.csv")
        if os.path.exists(target_path):
            logging.info(f"Training {target} prediction model from {target_path}")
            target_df = pd.read_csv(target_path)

            # Convert date column if present
            if "match_date" in target_df.columns:
                target_df["match_date"] = pd.to_datetime(target_df["match_date"])

            # Prepare data
            X_train, X_test, y_train, y_test, _ = prepare_data(target_df, target)

            # Train model
            model, metrics = train_transformer_model(
                X_train,
                X_test,
                y_train,
                y_test,
                is_classification=False,
                model_type=model_type,
                use_ensemble=use_ensemble,
                ensemble_type=ensemble_type,
                n_models=n_models,
                perform_hp_tuning=perform_hp_tuning,
                n_trials=n_trials,
                use_advanced_features=use_advanced_features,
            )

            # Model name with configuration
            model_name = f"{target}_transformer_{model_type}"
            if use_ensemble:
                model_name += f"_ensemble_{ensemble_type}"
            if perform_hp_tuning:
                model_name += "_tuned"
            if use_advanced_features:
                model_name += "_advanced"

            # Save model
            model_path = os.path.join(
                output_path, "expected_goals", f"{model_name}.joblib"
            )
            save_model(model, metrics, model_path)

            logging.info(f"Saved {target} model to {model_path}")
            logging.info(
                f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
            )


def train_events_models(data_path, output_path, model_config):
    """Train event prediction models with transformers.

    Args:
        data_path: Path to datasets
        output_path: Path to save models
        model_config: Model configuration dictionary
    """
    # Create output directory
    os.makedirs(os.path.join(output_path, "events"), exist_ok=True)

    # Get model configuration
    model_type = model_config.get("model_type", "basic")
    use_ensemble = model_config.get("use_ensemble", False)
    ensemble_type = model_config.get("ensemble_type", "stacking")
    n_models = model_config.get("n_models", 3)
    perform_hp_tuning = model_config.get("perform_hp_tuning", False)
    n_trials = model_config.get("n_trials", 50)
    use_advanced_features = model_config.get("use_advanced_features", False)

    # Events directory
    events_dir = os.path.join(data_path, "events")
    if not os.path.exists(events_dir):
        logging.warning(f"Events directory not found at {events_dir}")
        return

    # List event CSV files
    event_files = [f for f in os.listdir(events_dir) if f.endswith(".csv")]

    for event_file in event_files:
        event_path = os.path.join(events_dir, event_file)
        event_name = os.path.splitext(event_file)[0]

        logging.info(f"Training model for {event_name} from {event_path}")
        event_df = pd.read_csv(event_path)

        # Convert date column if present
        if "match_date" in event_df.columns:
            event_df["match_date"] = pd.to_datetime(event_df["match_date"])

        # Determine target column and task type
        if "both_teams_scored" in event_df.columns:
            target_col = "both_teams_scored"
            is_classification = True
        elif any(col.startswith("over_") for col in event_df.columns):
            # Find appropriate over/under column
            over_cols = [col for col in event_df.columns if col.startswith("over_")]
            if over_cols:
                target_col = over_cols[0]
                is_classification = True
            else:
                logging.warning(f"No target column found for {event_name}")
                continue
        else:
            # Use count column as target
            count_cols = [col for col in event_df.columns if col.endswith("_count")]
            event_type = event_name.split("_")[0]
            target_col = f"total_{event_type}"

            if target_col not in event_df.columns and count_cols:
                target_col = count_cols[0]

            if target_col not in event_df.columns:
                logging.warning(f"No target column found for {event_name}")
                continue

            is_classification = False

        # Prepare data
        X_train, X_test, y_train, y_test, _ = prepare_data(event_df, target_col)

        # Train model
        model, metrics = train_transformer_model(
            X_train,
            X_test,
            y_train,
            y_test,
            is_classification=is_classification,
            model_type=model_type,
            use_ensemble=use_ensemble,
            ensemble_type=ensemble_type,
            n_models=n_models,
            perform_hp_tuning=perform_hp_tuning,
            n_trials=n_trials,
            use_advanced_features=use_advanced_features,
        )

        # Model name with configuration
        model_name = f"{event_name}_transformer_{model_type}"
        if use_ensemble:
            model_name += f"_ensemble_{ensemble_type}"
        if perform_hp_tuning:
            model_name += "_tuned"
        if use_advanced_features:
            model_name += "_advanced"

        # Save model
        model_path = os.path.join(output_path, "events", f"{model_name}.joblib")
        save_model(model, metrics, model_path)

        if is_classification:
            logging.info(f"Saved {event_name} model to {model_path}")
            logging.info(
                f"Model metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
            )
        else:
            logging.info(f"Saved {event_name} model to {model_path}")
            logging.info(
                f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
            )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train enhanced transformer models for soccer match prediction"
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="datasets",
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="trained_models",
        help="Path to save trained models",
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        choices=["basic", "feature_tokenizer", "enhanced"],
        default="basic",
        help="Type of transformer model",
    )
    parser.add_argument(
        "--use-ensemble", action="store_true", help="Use ensemble modeling"
    )
    parser.add_argument(
        "--ensemble-type",
        type=str,
        choices=["voting", "stacking"],
        default="stacking",
        help="Type of ensemble",
    )
    parser.add_argument(
        "--n-models", type=int, default=3, help="Number of models in ensemble"
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of hyperparameter tuning trials",
    )
    parser.add_argument(
        "--use-advanced-features",
        action="store_true",
        help="Use advanced feature engineering",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["outcome", "events", "all"],
        default="all",
        help="Prediction task type to train models for",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Print GPU info if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    if device == "cuda":
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Create model configuration
    model_config = {
        "model_type": args.model_type,
        "use_ensemble": args.use_ensemble,
        "ensemble_type": args.ensemble_type,
        "n_models": args.n_models,
        "perform_hp_tuning": args.tune_hyperparameters,
        "n_trials": args.n_trials,
        "use_advanced_features": args.use_advanced_features,
    }

    # Train models
    if args.task in ["outcome", "all"]:
        train_match_outcome_models(args.data_path, args.output_path, model_config)

    if args.task in ["events", "all"]:
        train_events_models(args.data_path, args.output_path, model_config)

    logging.info("Model training completed successfully")


if __name__ == "__main__":
    main()
