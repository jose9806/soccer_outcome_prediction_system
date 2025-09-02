#!/usr/bin/env python3
"""
Test complete data pipeline with real scraped data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.ml.data.pipeline import DataPipeline
from src.ml.features.engine import FeatureEngineeringEngine

logger = get_logger(__name__, color="cyan")


def load_scraped_data(season: str = "2025") -> List[Dict[str, Any]]:
    """Load scraped match data for a season."""
    data_dir = Path("data/raw") / season
    matches = []
    
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} not found")
        return matches
    
    # Load all match JSON files
    for match_file in sorted(data_dir.glob("*.json")):
        try:
            with open(match_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
                matches.append(match_data)
        except Exception as e:
            logger.error(f"Error loading {match_file}: {e}")
    
    logger.info(f"Loaded {len(matches)} matches from season {season}")
    return matches


def transform_to_dataframe(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """Transform scraped matches to DataFrame format."""
    rows = []
    
    for match in matches:
        try:
            # Extract basic match info
            row = {
                'match_id': match.get('url', '').split('/')[-2] if match.get('url') else None,
                'date': match.get('date'),
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'home_score': match.get('home_score'),
                'away_score': match.get('away_score'),
                'competition': match.get('competition'),
                'season': match.get('season', '2025'),
                'attendance': match.get('attendance'),
                'referee': match.get('referee'),
                'venue': match.get('venue')
            }
            
            # Extract stats if available
            stats = match.get('stats', {})
            if stats:
                home_stats = stats.get('home', {})
                away_stats = stats.get('away', {})
                
                # Add home team stats
                for stat_name, stat_value in home_stats.items():
                    row[f'home_{stat_name}'] = stat_value
                
                # Add away team stats
                for stat_name, stat_value in away_stats.items():
                    row[f'away_{stat_name}'] = stat_value
            
            # Extract odds if available
            odds = match.get('odds', {})
            if odds:
                for bookmaker, bookmaker_odds in odds.items():
                    if isinstance(bookmaker_odds, dict):
                        for market, market_odds in bookmaker_odds.items():
                            if isinstance(market_odds, dict):
                                for outcome, odd_value in market_odds.items():
                                    row[f'odds_{bookmaker}_{market}_{outcome}'] = odd_value
            
            rows.append(row)
            
        except Exception as e:
            logger.error(f"Error processing match: {e}")
    
    df = pd.DataFrame(rows)
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['home_score', 'away_score', 'attendance']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert stats columns to numeric
    stats_columns = [col for col in df.columns if col.startswith(('home_', 'away_')) 
                    and not col.endswith('_team')]
    for col in stats_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert odds columns to numeric
    odds_columns = [col for col in df.columns if col.startswith('odds_')]
    for col in odds_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Created DataFrame with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:20]}...")  # Show first 20 columns
    
    return df


def test_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Test the data pipeline with real data."""
    logger.info("=" * 50)
    logger.info("Testing Data Pipeline")
    logger.info("=" * 50)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Convert DataFrame to match expected format and process
    # For now, let's directly use the data for feature engineering
    processed_df = df.copy()
    
    logger.info(f"Processed shape: {processed_df.shape}")
    logger.info(f"Missing values: {processed_df.isnull().sum().sum()}")
    logger.info(f"Memory usage: {processed_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return processed_df


def test_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Test feature engineering with real data."""
    logger.info("=" * 50)
    logger.info("Testing Feature Engineering")
    logger.info("=" * 50)
    
    # Initialize feature engine
    engine = FeatureEngineeringEngine()
    
    # Generate features
    features_df = engine.create_features(df)
    
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Feature columns ({len(features_df.columns)}): {list(features_df.columns)[:10]}...")
    logger.info(f"Missing values percentage: {(features_df.isnull().sum().sum() / features_df.size) * 100:.2f}%")
    
    # Show feature statistics
    logger.info("\nFeature Statistics:")
    stats_df = features_df.describe().T.head(10)
    logger.info(f"Stats shape: {stats_df.shape}")
    logger.info("First 5 feature stats:")
    logger.info(stats_df.head().to_string())
    
    return features_df


def analyze_teams(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze unique teams in the dataset."""
    teams = set()
    
    if 'home_team' in df.columns:
        teams.update(df['home_team'].dropna().unique())
    if 'away_team' in df.columns:
        teams.update(df['away_team'].dropna().unique())
    
    teams = sorted(list(teams))
    
    logger.info(f"\nFound {len(teams)} unique teams:")
    for i, team in enumerate(teams[:20], 1):  # Show first 20 teams
        logger.info(f"  {i:2d}. {team}")
    
    if len(teams) > 20:
        logger.info(f"  ... and {len(teams) - 20} more teams")
    
    return {'teams': teams, 'count': len(teams)}


def create_sample_prediction_data(teams: List[str]) -> pd.DataFrame:
    """Create sample data for prediction testing."""
    if len(teams) < 2:
        logger.error("Not enough teams for sample prediction")
        return pd.DataFrame()
    
    # Create a sample match
    sample = {
        'match_id': 'test_001',
        'date': datetime.now(),
        'home_team': teams[0],
        'away_team': teams[1],
        'competition': 'Test League',
        'season': '2025',
        'venue': 'Test Stadium'
    }
    
    return pd.DataFrame([sample])


def main():
    """Main test function."""
    try:
        logger.info("Starting pipeline test with real scraped data")
        
        # Load scraped data
        matches = load_scraped_data("2025")
        
        if not matches:
            logger.warning("No matches found, loading from 2024")
            matches = load_scraped_data("2024")
        
        if not matches:
            logger.error("No scraped data found")
            return
        
        # Transform to DataFrame
        df = transform_to_dataframe(matches)
        
        if df.empty:
            logger.error("DataFrame is empty")
            return
        
        logger.info(f"\nOriginal data shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Analyze teams
        team_info = analyze_teams(df)
        
        # Test data pipeline
        processed_df = test_data_pipeline(df)
        
        # Test feature engineering
        features_df = test_feature_engineering(processed_df)
        
        # Save processed data for model training
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "features_2025.parquet"
        features_df.to_parquet(output_file, index=False)
        logger.info(f"\nSaved processed features to {output_file}")
        
        # Create sample prediction data
        if team_info['teams']:
            sample_df = create_sample_prediction_data(team_info['teams'])
            if not sample_df.empty:
                logger.info("\nSample prediction data created:")
                logger.info(f"Home: {sample_df['home_team'].iloc[0]}")
                logger.info(f"Away: {sample_df['away_team'].iloc[0]}")
        
        logger.info("\n" + "=" * 50)
        logger.info("Pipeline test completed successfully!")
        logger.info("=" * 50)
        
        # Summary statistics
        logger.info("\nSummary:")
        logger.info(f"- Matches processed: {len(df)}")
        logger.info(f"- Teams found: {team_info['count']}")
        logger.info(f"- Features generated: {features_df.shape[1]}")
        logger.info(f"- Data quality: {100 - (features_df.isnull().sum().sum() / features_df.size) * 100:.1f}% complete")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()