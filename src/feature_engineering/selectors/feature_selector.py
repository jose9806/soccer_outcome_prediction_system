# src/feature_engineering/selectors/feature_selector.py
"""
Feature selector for identifying the most predictive features.

This module provides functionality for:
- Identifying the most important features for prediction
- Removing redundant or highly correlated features
- Applying dimensionality reduction techniques
- Evaluating feature importance
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select the most predictive features for soccer match prediction."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the feature selector.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        logger.info("Initialized FeatureSelector")

    def _prepare_data(
        self, df: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for feature selection by handling missing values and categorical features.

        Args:
            df: DataFrame with features
            target: Target variable name

        Returns:
            Tuple of (X_prepared, y_prepared)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for data preparation")
            return pd.DataFrame(), pd.Series()

        logger.info("Preparing data for feature selection")

        # Make a copy to avoid modifying the original
        data = df.copy()

        # Extract target variable
        if target not in data.columns:
            logger.error(f"Target variable '{target}' not found in DataFrame")
            return pd.DataFrame(), pd.Series()

        y = data[target].copy()
        X = data.drop(columns=[target])

        # Remove non-numeric columns and columns with excessive missing values
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        high_missing_cols = X.columns[
            X.isnull().mean() > 0.5
        ]  # Columns with >50% missing values

        cols_to_drop = list(non_numeric_cols) + list(high_missing_cols)

        if cols_to_drop:
            logger.info(
                f"Dropping {len(cols_to_drop)} non-numeric or high-missing columns"
            )
            X = X.drop(columns=cols_to_drop)

        # Encode target if it's categorical
        if not pd.api.types.is_numeric_dtype(y):
            logger.info(f"Encoding categorical target variable '{target}'")
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))

        logger.info(
            f"Prepared data: X with {X.shape[1]} features, y with {len(y)} samples"
        )
        return X, y

    def _get_feature_importance(
        self, X: pd.DataFrame, y: pd.Series, method: str = "random_forest"
    ) -> pd.DataFrame:
        """
        Calculate feature importance using the specified method.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Method to use ('random_forest', 'f_test', or 'mutual_info')

        Returns:
            DataFrame with feature importance scores
        """
        if X.empty or len(y) == 0:
            logger.warning("Empty data provided for feature importance calculation")
            return pd.DataFrame()

        logger.info(f"Calculating feature importance using method: {method}")

        # Check if the target is categorical or continuous
        is_classification = (
            len(y.unique()) < 10
        )  # Heuristic: if fewer than 10 unique values, treat as classification

        # Create imputer for missing values
        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Calculate feature importance based on the specified method
        if method == "random_forest":
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state
                )

            model.fit(X_imputed, y)
            importance = model.feature_importances_

        elif method == "f_test":
            if is_classification:
                selector = SelectKBest(f_classif, k="all")
            else:
                selector = SelectKBest(f_regression, k="all")

            selector.fit(X_imputed, y)
            importance = selector.scores_

        elif method == "mutual_info":
            if is_classification:
                selector = SelectKBest(mutual_info_classif, k="all")
            else:
                selector = SelectKBest(mutual_info_regression, k="all")

            selector.fit(X_imputed, y)
            importance = selector.scores_

        else:
            logger.error(f"Unknown feature importance method: {method}")
            return pd.DataFrame()

        # Create a DataFrame with feature importance
        importance_df = pd.DataFrame({"feature": X.columns, "importance": importance})

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        logger.info(f"Calculated importance for {len(importance_df)} features")
        return importance_df

    def _remove_correlated_features(
        self, X: pd.DataFrame, threshold: float = 0.9
    ) -> List[str]:
        """
        Identify and remove highly correlated features.

        Args:
            X: Feature DataFrame
            threshold: Correlation threshold above which to remove features

        Returns:
            List of features to keep (non-correlated)
        """
        if X.empty:
            logger.warning("Empty DataFrame provided for correlation analysis")
            return []

        logger.info(f"Removing correlated features with threshold {threshold}")

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Features to keep
        to_keep = [col for col in X.columns if col not in to_drop]

        logger.info(f"Identified {len(to_drop)} correlated features to remove")
        return to_keep

    def select_features(
        self, df: pd.DataFrame, target: str, max_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most predictive features for the target variable.

        Args:
            df: DataFrame with features and target
            target: Target variable to predict
            max_features: Maximum number of features to select

        Returns:
            Tuple of (DataFrame with selected features, List of selected feature names)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature selection")
            return pd.DataFrame(), []

        logger.info(f"Selecting features for target '{target}'")

        # Prepare data for feature selection
        X, y = self._prepare_data(df, target)

        if X.empty or len(y) == 0:
            logger.error("Data preparation failed, cannot select features")
            return df, list(df.columns)

        # Remove highly correlated features
        non_correlated_features = self._remove_correlated_features(X)
        X_non_correlated = X[non_correlated_features]

        logger.info(
            f"Remaining features after removing correlations: {len(non_correlated_features)}"
        )

        # Calculate feature importance
        importance_df = self._get_feature_importance(X_non_correlated, y)

        if importance_df.empty:
            logger.error("Feature importance calculation failed, using all features")
            return df, list(df.columns)

        # Determine how many features to keep
        if max_features is None:
            # Default: keep features with importance > mean importance
            mean_importance = importance_df["importance"].mean()
            selected_df = importance_df[importance_df["importance"] > mean_importance]
        else:
            # Keep top N features
            selected_df = importance_df.head(max_features)

        selected_features = selected_df["feature"].tolist()

        # Include the target in the selected features
        if target not in selected_features:
            selected_features.append(target)

        logger.info(f"Selected {len(selected_features)} features")

        # Return the DataFrame with only the selected features
        return df[selected_features], selected_features

    def apply_pca(
        self,
        df: pd.DataFrame,
        target: str,
        n_components: Optional[int] = None,
        explained_variance: float = 0.95,
    ) -> pd.DataFrame:
        """
        Apply Principal Component Analysis (PCA) for dimensionality reduction.

        Args:
            df: DataFrame with features
            target: Target variable name
            n_components: Number of components to keep, or None to use explained_variance
            explained_variance: Minimum explained variance to keep if n_components is None

        Returns:
            DataFrame with PCA features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for PCA")
            return pd.DataFrame()

        logger.info("Applying PCA for dimensionality reduction")

        # Prepare data for PCA
        X, y = self._prepare_data(df, target)

        if X.empty or len(y) == 0:
            logger.error("Data preparation failed, cannot apply PCA")
            return df

        # Create a pipeline with imputer, scaler, and PCA
        if n_components is None:
            # Use explained variance ratio
            pca = PCA(n_components=explained_variance, random_state=self.random_state)
            logger.info(
                f"Using PCA with explained variance ratio: {explained_variance}"
            )
        else:
            # Use fixed number of components
            pca = PCA(n_components=n_components, random_state=self.random_state)
            logger.info(f"Using PCA with {n_components} components")

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("pca", pca),
            ]
        )

        # Apply PCA
        X_pca = pipeline.fit_transform(X)

        # Create DataFrame with PCA components
        if n_components is None:
            # Get the actual number of components used
            n_components = pca.n_components_

        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns)

        # Add target variable back
        pca_df[target] = y.values

        logger.info(
            f"Applied PCA: reduced from {X.shape[1]} features to {len(pca_columns)} components"
        )
        return pca_df

    def rank_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Rank features by their importance for prediction.

        Args:
            df: DataFrame with features
            target: Target variable name

        Returns:
            DataFrame with ranked features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature ranking")
            return pd.DataFrame()

        logger.info(f"Ranking features for target '{target}'")

        # Prepare data for feature ranking
        X, y = self._prepare_data(df, target)

        if X.empty or len(y) == 0:
            logger.error("Data preparation failed, cannot rank features")
            return pd.DataFrame()

        # Use multiple methods for feature importance
        rf_importance = self._get_feature_importance(X, y, method="random_forest")
        f_test_importance = self._get_feature_importance(X, y, method="f_test")
        mi_importance = self._get_feature_importance(X, y, method="mutual_info")

        # Combine rankings from different methods
        ranking_df = pd.DataFrame({"feature": X.columns})

        # Add rankings from each method
        if not rf_importance.empty:
            ranking_df = pd.merge(
                ranking_df,
                rf_importance.rename(columns={"importance": "rf_importance"}),
                on="feature",
                how="left",
            )

        if not f_test_importance.empty:
            ranking_df = pd.merge(
                ranking_df,
                f_test_importance.rename(columns={"importance": "f_test_importance"}),
                on="feature",
                how="left",
            )

        if not mi_importance.empty:
            ranking_df = pd.merge(
                ranking_df,
                mi_importance.rename(columns={"importance": "mi_importance"}),
                on="feature",
                how="left",
            )

        # Calculate overall rank (average of ranks across methods)
        methods = [col for col in ranking_df.columns if col.endswith("_importance")]

        for method in methods:
            rank_col = method.replace("importance", "rank")
            ranking_df[rank_col] = ranking_df[method].rank(ascending=False)

        rank_cols = [col for col in ranking_df.columns if col.endswith("_rank")]

        if rank_cols:
            ranking_df["avg_rank"] = ranking_df[rank_cols].mean(axis=1)
            ranking_df["overall_rank"] = ranking_df["avg_rank"].rank()

            # Sort by overall rank
            ranking_df = ranking_df.sort_values("overall_rank")

        logger.info(f"Ranked {len(ranking_df)} features")
        return ranking_df
