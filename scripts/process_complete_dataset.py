#!/usr/bin/env python3
"""
Complete Dataset Processing Pipeline with Temporal Weighting.

Processes all 1020+ JSON files with temporal weighting where recent data
is more valuable. Saves to proper data/features/ structure.
"""

import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
import time
from tqdm import tqdm
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src.ml.features.engine import FeatureEngineeringEngine, FeatureConfig
from src.config.logging_config import get_logger


class TemporalWeighting:
    """
    Temporal weighting system that gives more importance to recent data.
    
    Uses exponential decay: weight = exp(-λ * days_ago)
    """
    
    def __init__(self, decay_rate: float = 0.001, min_weight: float = 0.1):
        """
        Initialize temporal weighting.
        
        Args:
            decay_rate: Lambda parameter for exponential decay (higher = faster decay)
            min_weight: Minimum weight for very old data
        """
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.logger = get_logger("TemporalWeighting")
        
    def calculate_weights(self, dates: pd.Series, reference_date: datetime = None) -> pd.Series:
        """Calculate temporal weights for given dates."""
        if reference_date is None:
            reference_date = datetime.now()
            
        # Convert to datetime if needed
        if not isinstance(dates.iloc[0], datetime):
            dates = pd.to_datetime(dates)
            
        # Calculate days ago
        days_ago = (reference_date - dates).dt.days
        
        # Calculate exponential weights
        weights = np.exp(-self.decay_rate * days_ago)
        
        # Apply minimum weight
        weights = np.maximum(weights, self.min_weight)
        
        return weights
    
    def get_weighting_info(self, dates: pd.Series) -> dict:
        """Get information about the weighting scheme."""
        if not isinstance(dates.iloc[0], datetime):
            dates = pd.to_datetime(dates)
            
        reference_date = datetime.now()
        weights = self.calculate_weights(dates, reference_date)
        
        return {
            'reference_date': reference_date.isoformat(),
            'date_range': f"{dates.min()} to {dates.max()}",
            'decay_rate': self.decay_rate,
            'min_weight': self.min_weight,
            'weight_range': f"{weights.min():.4f} to {weights.max():.4f}",
            'avg_weight': weights.mean(),
            'recent_data_ratio': (weights > 0.8).sum() / len(weights)
        }


class ComprehensiveDataLoader:
    """
    Comprehensive data loader for all JSON files with quality validation.
    """
    
    def __init__(self, temporal_weighting: TemporalWeighting = None):
        self.temporal_weighting = temporal_weighting or TemporalWeighting()
        self.logger = get_logger("ComprehensiveDataLoader")
        
    def load_all_data(self) -> pd.DataFrame:
        """Load all JSON files with comprehensive processing."""
        self.logger.info("🚀 Starting comprehensive data loading...")
        
        # Find all JSON files
        json_files = sorted(glob.glob("data/raw/**/*.json", recursive=True))
        self.logger.info(f"Found {len(json_files)} JSON files")
        
        # Analyze file distribution
        self._analyze_file_distribution(json_files)
        
        # Load all files
        matches = []
        failed_files = []
        
        with tqdm(json_files, desc="Loading JSON files", unit="files") as pbar:
            for file_path in pbar:
                try:
                    match_data = self._load_single_file(file_path)
                    if match_data:
                        matches.append(match_data)
                    
                    # Update progress description
                    pbar.set_description(f"Loaded {len(matches)} matches")
                    
                except Exception as e:
                    failed_files.append((file_path, str(e)))
                    if len(failed_files) <= 10:  # Log first 10 errors
                        self.logger.warning(f"Failed to load {file_path}: {e}")
        
        if len(failed_files) > 10:
            self.logger.warning(f"...and {len(failed_files) - 10} more files failed")
        
        if not matches:
            raise ValueError("No valid matches loaded")
        
        # Convert to DataFrame
        self.logger.info("Converting to DataFrame and processing...")
        df = pd.DataFrame(matches)
        
        # Clean and process data
        df = self._clean_and_process_data(df)
        
        # Add temporal weights
        df = self._add_temporal_weights(df)
        
        # Final validation
        self._validate_final_dataset(df)
        
        return df
    
    def _analyze_file_distribution(self, json_files: List[str]) -> None:
        """Analyze the distribution of files across years."""
        years = {}
        for file_path in json_files:
            year = Path(file_path).parts[2]  # data/raw/YEAR/file.json
            years[year] = years.get(year, 0) + 1
        
        self.logger.info("📊 File distribution by year:")
        for year in sorted(years.keys()):
            self.logger.info(f"   {year}: {years[year]} files")
    
    def _load_single_file(self, file_path: str) -> dict:
        """Load and extract data from a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract essential fields
            match_data = {
                # Basic match information
                'match_id': data.get('match_id', ''),
                'date': data.get('date', ''),
                'home_team': data.get('home_team', ''),
                'away_team': data.get('away_team', ''),
                'home_score': data.get('home_score', 0),
                'away_score': data.get('away_score', 0),
                'season': data.get('season', ''),
                'competition': data.get('competition', ''),
                'stage': data.get('stage', ''),
                'status': data.get('status', ''),
                
                # Derived fields
                'result': self._determine_result(data.get('home_score', 0), data.get('away_score', 0)),
                'total_goals': data.get('home_score', 0) + data.get('away_score', 0),
                'both_teams_scored': (data.get('home_score', 0) > 0 and data.get('away_score', 0) > 0),
                'goal_difference': data.get('home_score', 0) - data.get('away_score', 0),
                
                # Full-time statistics
                'possession_home': self._safe_extract(data, ['full_time_stats', 'possession', 'home'], 50),
                'possession_away': self._safe_extract(data, ['full_time_stats', 'possession', 'away'], 50),
                'shots_on_target_home': self._safe_extract(data, ['full_time_stats', 'shots_on_goal', 'home'], 0),
                'shots_on_target_away': self._safe_extract(data, ['full_time_stats', 'shots_on_goal', 'away'], 0),
                'shots_off_target_home': self._safe_extract(data, ['full_time_stats', 'shots_off_goal', 'home'], 0),
                'shots_off_target_away': self._safe_extract(data, ['full_time_stats', 'shots_off_goal', 'away'], 0),
                'total_shots_home': self._safe_extract(data, ['full_time_stats', 'goal_attempts', 'home'], 0),
                'total_shots_away': self._safe_extract(data, ['full_time_stats', 'goal_attempts', 'away'], 0),
                'blocked_shots_home': self._safe_extract(data, ['full_time_stats', 'blocked_shots', 'home'], 0),
                'blocked_shots_away': self._safe_extract(data, ['full_time_stats', 'blocked_shots', 'away'], 0),
                'corners_home': self._safe_extract(data, ['full_time_stats', 'corner_kicks', 'home'], 0),
                'corners_away': self._safe_extract(data, ['full_time_stats', 'corner_kicks', 'away'], 0),
                'fouls_home': self._safe_extract(data, ['full_time_stats', 'fouls', 'home'], 0),
                'fouls_away': self._safe_extract(data, ['full_time_stats', 'fouls', 'away'], 0),
                'yellow_cards_home': self._safe_extract(data, ['full_time_stats', 'yellow_cards', 'home'], 0),
                'yellow_cards_away': self._safe_extract(data, ['full_time_stats', 'yellow_cards', 'away'], 0),
                'red_cards_home': self._safe_extract(data, ['full_time_stats', 'red_cards', 'home'], 0),
                'red_cards_away': self._safe_extract(data, ['full_time_stats', 'red_cards', 'away'], 0),
                'goalkeeper_saves_home': self._safe_extract(data, ['full_time_stats', 'goalkeeper_saves', 'home'], 0),
                'goalkeeper_saves_away': self._safe_extract(data, ['full_time_stats', 'goalkeeper_saves', 'away'], 0),
                'offsides_home': self._safe_extract(data, ['full_time_stats', 'offsides', 'home'], 0),
                'offsides_away': self._safe_extract(data, ['full_time_stats', 'offsides', 'away'], 0),
                
                # First half statistics
                'possession_1h_home': self._safe_extract(data, ['first_half_stats', 'possession', 'home'], None),
                'possession_1h_away': self._safe_extract(data, ['first_half_stats', 'possession', 'away'], None),
                'shots_on_target_1h_home': self._safe_extract(data, ['first_half_stats', 'shots_on_goal', 'home'], None),
                'shots_on_target_1h_away': self._safe_extract(data, ['first_half_stats', 'shots_on_goal', 'away'], None),
                
                # Second half statistics  
                'possession_2h_home': self._safe_extract(data, ['second_half_stats', 'possession', 'home'], None),
                'possession_2h_away': self._safe_extract(data, ['second_half_stats', 'possession', 'away'], None),
                'shots_on_target_2h_home': self._safe_extract(data, ['second_half_stats', 'shots_on_goal', 'home'], None),
                'shots_on_target_2h_away': self._safe_extract(data, ['second_half_stats', 'shots_on_goal', 'away'], None),
                
                # Betting odds (average across bookmakers)
                'odds_home_win': None,
                'odds_draw': None,
                'odds_away_win': None,
                'odds_over_2_5': None,
                'odds_under_2_5': None,
                'implied_prob_home': None,
                'implied_prob_draw': None,
                'implied_prob_away': None,
                
                # Metadata
                'file_path': file_path,
                'year': Path(file_path).parts[2]
            }
            
            # Process odds
            self._extract_odds_data(data, match_data)
            
            return match_data
            
        except Exception as e:
            raise Exception(f"Error processing {file_path}: {e}")
    
    def _safe_extract(self, data: dict, path: List[str], default=None):
        """Safely extract nested dictionary values."""
        try:
            value = data
            for key in path:
                value = value[key]
            return value if value is not None else default
        except (KeyError, TypeError):
            return default
    
    def _determine_result(self, home_score: int, away_score: int) -> str:
        """Determine match result from home team perspective."""
        if home_score > away_score:
            return 'H'  # Home win
        elif home_score < away_score:
            return 'A'  # Away win
        else:
            return 'D'  # Draw
    
    def _extract_odds_data(self, data: dict, match_data: dict) -> None:
        """Extract and process betting odds data."""
        odds_data = data.get('odds', {})
        
        # Match winner odds
        match_winner_odds = odds_data.get('match_winner', [])
        if match_winner_odds:
            home_odds = [float(o.get('home_win', 0)) for o in match_winner_odds if o.get('home_win')]
            draw_odds = [float(o.get('draw', 0)) for o in match_winner_odds if o.get('draw')]
            away_odds = [float(o.get('away_win', 0)) for o in match_winner_odds if o.get('away_win')]
            
            if home_odds:
                match_data['odds_home_win'] = np.mean(home_odds)
                match_data['implied_prob_home'] = 1 / np.mean(home_odds)
            if draw_odds:
                match_data['odds_draw'] = np.mean(draw_odds)
                match_data['implied_prob_draw'] = 1 / np.mean(draw_odds)
            if away_odds:
                match_data['odds_away_win'] = np.mean(away_odds)
                match_data['implied_prob_away'] = 1 / np.mean(away_odds)
        
        # Over/Under 2.5 goals odds
        over_under_odds = odds_data.get('over_under_2_5', [])
        if over_under_odds:
            over_odds = [float(o.get('over', 0)) for o in over_under_odds if o.get('over')]
            under_odds = [float(o.get('under', 0)) for o in over_under_odds if o.get('under')]
            
            if over_odds:
                match_data['odds_over_2_5'] = np.mean(over_odds)
            if under_odds:
                match_data['odds_under_2_5'] = np.mean(under_odds)
    
    def _clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the loaded data."""
        self.logger.info("🧹 Cleaning and processing data...")
        
        initial_size = len(df)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Remove rows with missing essential data
        essential_columns = ['home_team', 'away_team', 'home_score', 'away_score']
        df = df.dropna(subset=essential_columns)
        
        # Filter out invalid scores (negative scores)
        df = df[(df['home_score'] >= 0) & (df['away_score'] >= 0)]
        
        # Sort by date for temporal processing
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add derived statistics
        df['shot_accuracy_home'] = np.where(
            df['total_shots_home'] > 0, 
            df['shots_on_target_home'] / df['total_shots_home'], 
            0
        )
        df['shot_accuracy_away'] = np.where(
            df['total_shots_away'] > 0,
            df['shots_on_target_away'] / df['total_shots_away'],
            0
        )
        
        df['conversion_rate_home'] = np.where(
            df['shots_on_target_home'] > 0,
            df['home_score'] / df['shots_on_target_home'],
            0
        )
        df['conversion_rate_away'] = np.where(
            df['shots_on_target_away'] > 0,
            df['away_score'] / df['shots_on_target_away'], 
            0
        )
        
        # Goal difference statistics
        df['abs_goal_difference'] = df['goal_difference'].abs()
        df['is_high_scoring'] = df['total_goals'] >= 3
        df['is_low_scoring'] = df['total_goals'] <= 1
        df['margin_of_victory'] = np.where(df['result'] != 'D', df['abs_goal_difference'], 0)
        
        final_size = len(df)
        removed = initial_size - final_size
        
        self.logger.info(f"Data cleaning completed:")
        self.logger.info(f"   Initial size: {initial_size}")
        self.logger.info(f"   Final size: {final_size}")  
        self.logger.info(f"   Removed: {removed} ({100*removed/initial_size:.1f}%)")
        
        return df
    
    def _add_temporal_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal weights to the dataset."""
        self.logger.info("⚖️ Adding temporal weights...")
        
        # Calculate weights
        weights = self.temporal_weighting.calculate_weights(df['date'])
        df['temporal_weight'] = weights
        
        # Add weight-based features
        df['is_recent'] = weights > 0.8  # Recent data (high weight)
        df['is_historical'] = weights < 0.3  # Historical data (low weight)
        df['weight_quartile'] = pd.qcut(weights, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Log weighting info
        weighting_info = self.temporal_weighting.get_weighting_info(df['date'])
        self.logger.info("📊 Temporal weighting summary:")
        for key, value in weighting_info.items():
            self.logger.info(f"   {key}: {value}")
        
        return df
    
    def _validate_final_dataset(self, df: pd.DataFrame) -> None:
        """Validate the final processed dataset."""
        self.logger.info("✅ Validating final dataset...")
        
        # Basic validation
        assert len(df) > 0, "Dataset is empty"
        assert 'date' in df.columns, "Date column missing"
        assert df['date'].dtype == '<M8[ns]', "Date column not properly converted"
        
        # Temporal validation
        date_range = df['date'].max() - df['date'].min()
        self.logger.info(f"📅 Temporal coverage: {date_range.days} days")
        self.logger.info(f"   From: {df['date'].min()}")
        self.logger.info(f"   To: {df['date'].max()}")
        
        # Data quality metrics
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        self.logger.info(f"📊 Data quality metrics:")
        self.logger.info(f"   Shape: {df.shape}")
        self.logger.info(f"   Missing values: {missing_cells}/{total_cells} ({missing_percentage:.2f}%)")
        self.logger.info(f"   Unique matches: {df['match_id'].nunique()}")
        self.logger.info(f"   Unique teams: {pd.concat([df['home_team'], df['away_team']]).nunique()}")
        self.logger.info(f"   Competitions: {df['competition'].nunique()}")
        
        # Result distribution
        result_dist = df['result'].value_counts(normalize=True)
        self.logger.info(f"🎯 Result distribution:")
        for result, pct in result_dist.items():
            self.logger.info(f"   {result}: {pct:.3f}")
        
        # Temporal weight distribution
        weight_stats = df['temporal_weight'].describe()
        self.logger.info(f"⚖️ Temporal weight distribution:")
        for stat, value in weight_stats.items():
            self.logger.info(f"   {stat}: {value:.4f}")


def process_complete_dataset():
    """Main function to process the complete dataset."""
    logger = get_logger("CompleteDatasetProcessor")
    
    logger.info("🚀 STARTING COMPLETE DATASET PROCESSING")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize temporal weighting (recent data valued more)
        temporal_weighting = TemporalWeighting(
            decay_rate=0.0005,  # Moderate decay - data loses half value in ~1400 days
            min_weight=0.1      # Old data still has some value
        )
        
        # Load complete dataset
        loader = ComprehensiveDataLoader(temporal_weighting)
        df = loader.load_all_data()
        
        # Create feature engineering configuration
        logger.info("🎯 Starting feature engineering...")
        config = FeatureConfig(
            enable_temporal=True,
            enable_contextual=True,
            enable_advanced=True,
            enable_selection=True,
            performance_monitoring=True
        )
        
        # Initialize and run feature engineering
        engine = FeatureEngineeringEngine(config)
        df_features = engine.create_features(df)
        
        # Save processed dataset
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        
        # Save main dataset
        main_file = f"data/features/processed_features_complete_{timestamp}.csv"
        df_features.to_csv(main_file, index=False)
        logger.info(f"💾 Main dataset saved: {main_file}")
        
        # Save temporal weighted version
        # Apply weights to numeric features for training
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'temporal_weight']
        
        df_weighted = df_features.copy()
        for col in numeric_cols:
            if col in df_weighted.columns:
                df_weighted[f"{col}_weighted"] = df_weighted[col] * df_weighted['temporal_weight']
        
        weighted_file = f"data/features/processed_features_weighted_{timestamp}.csv"
        df_weighted.to_csv(weighted_file, index=False)
        logger.info(f"💾 Weighted dataset saved: {weighted_file}")
        
        # Save metadata
        feature_names = engine.get_feature_names()
        feature_importance = engine.get_feature_importance()
        processing_report = engine.get_processing_report()
        weighting_info = temporal_weighting.get_weighting_info(df['date'])
        
        metadata = {
            'processing_timestamp': timestamp,
            'dataset_info': {
                'total_matches': len(df_features),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'total_features': len(feature_names),
                'competitions': df['competition'].nunique(),
                'teams': pd.concat([df['home_team'], df['away_team']]).nunique(),
                'years_covered': sorted(df['year'].unique())
            },
            'temporal_weighting': weighting_info,
            'feature_engineering': processing_report,
            'feature_names': feature_names,
            'data_quality': {
                'missing_percentage': float((df_features.isnull().sum().sum() / (df_features.shape[0] * df_features.shape[1])) * 100),
                'result_distribution': df['result'].value_counts(normalize=True).to_dict(),
                'temporal_coverage_days': int((df['date'].max() - df['date'].min()).days)
            }
        }
        
        metadata_file = f"data/features/dataset_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"📋 Metadata saved: {metadata_file}")
        
        # Generate summary report
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("🎉 DATASET PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📊 FINAL RESULTS:")
        logger.info(f"   Total matches processed: {len(df_features):,}")
        logger.info(f"   Total features created: {len(feature_names):,}")
        logger.info(f"   Processing time: {processing_time:.1f} seconds")
        logger.info(f"   Processing speed: {len(df_features)/processing_time:.1f} matches/second")
        logger.info(f"   Data coverage: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"   Missing data: {metadata['data_quality']['missing_percentage']:.2f}%")
        
        logger.info(f"\n📁 FILES CREATED:")
        logger.info(f"   Main dataset: {main_file}")
        logger.info(f"   Weighted dataset: {weighted_file}") 
        logger.info(f"   Metadata: {metadata_file}")
        
        logger.info(f"\n🎯 READY FOR ML MODEL TRAINING!")
        logger.info("="*60)
        
        return df_features, metadata
        
    except Exception as e:
        logger.error(f"❌ Dataset processing failed: {e}")
        raise


if __name__ == "__main__":
    try:
        df, metadata = process_complete_dataset()
        print("\n✅ Processing completed successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Features created: {len(metadata['feature_names'])}")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)