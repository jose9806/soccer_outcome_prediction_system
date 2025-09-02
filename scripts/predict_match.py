#!/usr/bin/env python3
"""
Soccer Match Prediction Script

Predicts match outcomes using only team names as input.
Automatically generates features and provides betting recommendations.

Usage:
    python scripts/predict_match.py "Real Madrid" "Barcelona"
    python scripts/predict_match.py --home "Liverpool" --away "Manchester City"
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.prediction.match_predictor import MatchPredictor
from src.config.logging_config import get_logger


def main():
    """Main prediction interface."""
    parser = argparse.ArgumentParser(
        description='Predict soccer match outcome using team names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_match.py "Real Madrid" "Barcelona"
  python scripts/predict_match.py --home "Liverpool" --away "Manchester City"
  python scripts/predict_match.py "Dep Cali" "Bucaramanga" --date "2025-02-14"
        """
    )
    
    # Team arguments
    parser.add_argument('home_team', nargs='?', help='Home team name')
    parser.add_argument('away_team', nargs='?', help='Away team name')
    parser.add_argument('--home', help='Home team name (alternative)')
    parser.add_argument('--away', help='Away team name (alternative)')
    
    # Optional parameters
    parser.add_argument('--date', help='Match date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--competition', default='', help='Competition name')
    parser.add_argument('--models-dir', default='data/models', help='Directory containing trained models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Determine team names
    home_team = args.home_team or args.home
    away_team = args.away_team or args.away
    
    if not home_team or not away_team:
        parser.error("Must specify both home and away team names")
    
    # Setup logging
    logger = get_logger("MatchPredictor")
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse match date
    match_date = datetime.now()
    if args.date:
        try:
            match_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1
    
    try:
        # Initialize predictor
        logger.info(f"🔮 Initializing match predictor...")
        predictor = MatchPredictor(models_dir=args.models_dir)
        
        # Make prediction
        logger.info(f"⚽ Predicting: {home_team} vs {away_team}")
        if args.date:
            logger.info(f"📅 Match Date: {match_date.strftime('%Y-%m-%d')}")
        
        prediction = predictor.predict_match(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            competition=args.competition
        )
        
        if args.json:
            # JSON output for programmatic use
            import json
            print(json.dumps(prediction.to_dict(), indent=2, default=str))
        else:
            # Human-readable output
            print_prediction_summary(prediction, home_team, away_team)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_prediction_summary(prediction, home_team: str, away_team: str):
    """Print human-readable prediction summary."""
    print("\n" + "="*60)
    print(f"🏆 MATCH PREDICTION: {home_team} vs {away_team}")
    print("="*60)
    
    # Main prediction
    print(f"\n🎯 PREDICTED OUTCOME: {prediction.predicted_outcome}")
    print(f"🎲 CONFIDENCE: {prediction.confidence:.1%}")
    
    # Probabilities
    print(f"\n📊 PROBABILITIES:")
    for outcome, prob in prediction.probabilities.items():
        outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
        bar = "█" * int(prob * 20)
        print(f"   {outcome_name:10s}: {prob:.1%} {bar}")
    
    # Additional predictions
    if prediction.expected_goals_home is not None:
        print(f"\n⚽ EXPECTED GOALS:")
        print(f"   {home_team}: {prediction.expected_goals_home:.2f}")
        print(f"   {away_team}: {prediction.expected_goals_away:.2f}")
    
    if prediction.over_under_2_5 is not None:
        print(f"\n📈 SPECIAL MARKETS:")
        print(f"   Over 2.5 Goals: {prediction.over_under_2_5:.1%}")
        
    if prediction.both_teams_score is not None:
        print(f"   Both Teams Score: {prediction.both_teams_score:.1%}")
    
    # Model info
    print(f"\n🤖 MODEL INFO:")
    print(f"   Model: {prediction.model_name}")
    print(f"   Version: {prediction.model_version}")
    print(f"   Features Used: {len(prediction.features_used)}")
    
    # Uncertainty measure
    entropy = prediction.entropy
    uncertainty = "Low" if entropy < 1.0 else "Medium" if entropy < 1.5 else "High"
    print(f"   Uncertainty: {uncertainty} ({entropy:.2f})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    sys.exit(main())