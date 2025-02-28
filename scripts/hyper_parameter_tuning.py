"""
Hyperparameter tuning for soccer match prediction models.

This script performs systematic hyperparameter optimization for various model types.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import uniform, randint, loguniform


def load_dataset(
    data_path: str, dataset_type: str, target_name: str
) -> Tuple[pd.DataFrame, str]:
    """Load dataset for a specific prediction task.

    Args:
        data_path: Base path to datasets
        dataset_type: Type of dataset (match_outcome or events)
        target_name: Name of the target dataset

    Returns:
        DataFrame with the dataset and target column name
    """
    # Determine full path
    if dataset_type == "match_outcome":
        # For match outcome datasets
        if target_name == "outcome":
            filepath = os.path.join(
                data_path, "match_outcome", "match_outcome_3way.csv"
            )
            target_col = "outcome_num"
        else:
            # For expected goals datasets
            filepath = os.path.join(data_path, "match_outcome", f"{target_name}.csv")
            target_col = target_name
    else:
        # For event datasets
        filepath = os.path.join(data_path, "events", f"{target_name}.csv")

        # Determine target column
        df = pd.read_csv(filepath)
        if "both_teams_scored" in df.columns:
            target_col = "both_teams_scored"
        elif any(col.startswith("over_") for col in df.columns):
            target_col = [col for col in df.columns if col.startswith("over_")][0]
        else:
            event_type = target_name.split("_")[0]
            target_col = f"total_{event_type}"

            if target_col not in df.columns:
                raise ValueError(f"Could not determine target column for {target_name}")

        return df, target_col

    # Load dataset
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    return pd.read_csv(filepath), target_col


def prepare_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for model training.

    Args:
        df: Input DataFrame
        target_col: Target column name

    Returns:
        Tuple of (X, y)
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
    X = X.drop(columns=[col for col in non_feature_cols if col in X.columns])

    return X, y


def create_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for the data.

    Args:
        X: Feature DataFrame

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

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


def get_parameter_grid(model_type: str, is_classification: bool) -> Dict:
    """Get hyperparameter grid for a specific model type.

    Args:
        model_type: Type of model (lgb, xgb)
        is_classification: Whether this is a classification task

    Returns:
        Dictionary with hyperparameter grid
    """
    if model_type == "lgb":
        # LightGBM parameters
        if is_classification:
            param_grid = {
                "model__n_estimators": randint(50, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__num_leaves": randint(20, 150),
                "model__max_depth": randint(3, 12),
                "model__min_child_samples": randint(10, 100),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__reg_alpha": loguniform(1e-3, 10),
                "model__reg_lambda": loguniform(1e-3, 10),
            }
        else:
            param_grid = {
                "model__n_estimators": randint(50, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__num_leaves": randint(20, 150),
                "model__max_depth": randint(3, 12),
                "model__min_child_samples": randint(10, 100),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__reg_alpha": loguniform(1e-3, 10),
                "model__reg_lambda": loguniform(1e-3, 10),
            }
    elif model_type == "xgb":
        # XGBoost parameters
        if is_classification:
            param_grid = {
                "model__n_estimators": randint(50, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(3, 12),
                "model__min_child_weight": randint(1, 10),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__gamma": loguniform(1e-3, 1),
                "model__reg_alpha": loguniform(1e-3, 10),
                "model__reg_lambda": loguniform(1e-3, 10),
            }
        else:
            param_grid = {
                "model__n_estimators": randint(50, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(3, 12),
                "model__min_child_weight": randint(1, 10),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__gamma": loguniform(1e-3, 1),
                "model__reg_alpha": loguniform(1e-3, 10),
                "model__reg_lambda": loguniform(1e-3, 10),
            }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return param_grid


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    is_classification: bool,
    n_iter: int = 20,
    cv: int = 5,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict]:
    """Tune hyperparameters for a model.

    Args:
        X: Feature DataFrame
        y: Target series
        model_type: Type of model (lgb, xgb)
        is_classification: Whether this is a classification task
        n_iter: Number of parameter settings to try
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel

    Returns:
        Tuple of (best_pipeline, best_params)
    """
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X)

    # Create base model
    if model_type == "lgb":
        if is_classification:
            model = lgb.LGBMClassifier(
                objective="multiclass" if len(np.unique(y)) > 2 else "binary",
                random_state=42,
            )
        else:
            model = lgb.LGBMRegressor(objective="regression", random_state=42)
    elif model_type == "xgb":
        if is_classification:
            model = xgb.XGBClassifier(
                objective=(
                    "multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic"
                ),
                random_state=42,
            )
        else:
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Create parameter grid
    param_grid = get_parameter_grid(model_type, is_classification)

    # Create scorer
    if is_classification:
        if len(np.unique(y)) > 2:  # Multi-class
            scorer = make_scorer(f1_score, average="weighted")
        else:  # Binary
            scorer = make_scorer(f1_score)
    else:
        # For regression, use negative mean squared error (to maximize)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Create random search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=2,
        random_state=42,
    )

    # Fit random search
    logging.info(f"Starting hyperparameter tuning with {n_iter} iterations...")
    random_search.fit(X, y)

    # Get best parameters
    logging.info(f"Best score: {random_search.best_score_:.4f}")
    logging.info(f"Best parameters: {random_search.best_params_}")

    return random_search.best_estimator_, random_search.best_params_


def save_model_and_params(model: Pipeline, params: Dict, filepath: str):
    """Save model and parameters.

    Args:
        model: Trained model
        params: Hyperparameters
        filepath: Path to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model
    joblib.dump(model, filepath)

    # Save parameters
    params_path = f"{os.path.splitext(filepath)[0]}_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    logging.info(f"Saved model to {filepath}")
    logging.info(f"Saved parameters to {params_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for match prediction models"
    )
    parser.add_argument(
        "--data-path", type=str, default="datasets", help="Path to datasets directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="trained_models/tuned",
        help="Path to save tuned models",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lgb", "xgb"],
        default="lgb",
        help="Type of model to tune",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["match_outcome", "events"],
        default="match_outcome",
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target dataset name (e.g., outcome, home_goals, yellow_cards_over_3)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=20, help="Number of parameter settings to try"
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of jobs to run in parallel"
    )
    parser.add_argument(
        "--log-level",
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

    # Load dataset
    logging.info(f"Loading {args.dataset_type} dataset for {args.target}...")
    df, target_col = load_dataset(args.data_path, args.dataset_type, args.target)

    # Prepare data
    logging.info("Preparing data...")
    X, y = prepare_data(df, target_col)

    # Determine if classification task
    is_classification = pd.api.types.is_integer_dtype(y) and len(np.unique(y)) < 10
    task_type = "classification" if is_classification else "regression"
    logging.info(f"Task type: {task_type}")

    # Tune model
    logging.info(f"Tuning {args.model_type} model...")
    model, params = tune_model(
        X,
        y,
        args.model_type,
        is_classification,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
    )

    # Save model and parameters
    output_subdir = "outcome" if args.dataset_type == "match_outcome" else "events"
    if args.target in ["home_goals", "away_goals", "total_goals"]:
        output_subdir = "expected_goals"

    output_filename = f"{args.target}_{args.model_type}_tuned.joblib"
    output_path = os.path.join(args.output_path, output_subdir, output_filename)

    save_model_and_params(model, params, output_path)

    logging.info("Hyperparameter tuning completed successfully")


if __name__ == "__main__":
    main()
