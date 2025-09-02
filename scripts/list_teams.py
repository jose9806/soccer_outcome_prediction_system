#!/usr/bin/env python3
"""
List Available Teams Script

Shows all teams available in the historical data for predictions.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.prediction.match_predictor import MatchPredictor
from src.config.logging_config import get_logger


def main():
    """Main function to list teams."""
    parser = argparse.ArgumentParser(description='List available teams for prediction')
    parser.add_argument('--search', '-s', help='Search for teams containing this text')
    parser.add_argument('--count', '-c', action='store_true', help='Show team count only')
    parser.add_argument('--data-dir', default='data/raw', help='Data directory path')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor (loads team data)
        print("📊 Loading team data...")
        predictor = MatchPredictor(data_dir=args.data_dir)
        
        # Get teams
        teams = predictor.get_available_teams()
        
        if args.search:
            # Search for teams
            suggestions = predictor.get_team_suggestions(args.search)
            if suggestions:
                print(f"\n🔍 Teams matching '{args.search}':")
                for i, team in enumerate(suggestions, 1):
                    print(f"  {i:2d}. {team}")
            else:
                print(f"❌ No teams found matching '{args.search}'")
        
        elif args.count:
            print(f"\n📈 Total teams available: {len(teams)}")
        
        else:
            # List all teams
            print(f"\n⚽ Available Teams ({len(teams)} total):")
            print("=" * 50)
            
            for i, team in enumerate(teams, 1):
                print(f"  {i:3d}. {team}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())