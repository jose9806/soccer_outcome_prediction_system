#!/usr/bin/env python3
"""
Complete ML Pipeline Runner

Runs the complete machine learning pipeline from data processing to model training.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger


def run_command(cmd: list, description: str, logger) -> bool:
    """Run a command and return success status."""
    logger.info(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        
        logger.info(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def main():
    """Run the complete ML pipeline."""
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument('--skip-processing', action='store_true', 
                       help='Skip data processing (use existing processed data)')
    parser.add_argument('--optimize', action='store_true',
                       help='Enable hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=15,
                       help='Number of optimization trials')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    logger = get_logger("FullPipeline")
    
    print("🚀 STARTING COMPLETE ML PIPELINE")
    print("=" * 60)
    
    start_time = datetime.now()
    success = True
    
    # Step 1: Data Processing
    if not args.skip_processing:
        cmd = ["python", "scripts/process_complete_dataset.py"]
        if not run_command(cmd, "Data processing and feature engineering", logger):
            success = False
    else:
        logger.info("⏭️ Skipping data processing (using existing data)")
    
    # Step 2: Model Training
    if success:
        cmd = ["python", "scripts/train_ml_models.py", "--cv-folds", str(args.cv_folds)]
        
        if args.optimize:
            cmd.extend(["--optimize", "--trials", str(args.trials)])
        
        if not run_command(cmd, "Model training and evaluation", logger):
            success = False
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"✅ Total pipeline duration: {duration}")
        
        # Show next steps
        print("\n💡 Next steps:")
        print("   • Test predictions: python scripts/predict_match.py 'Team A' 'Team B'")
        print("   • List teams: python scripts/list_teams.py")
        print("   • Check models: ls -la data/models/")
        
    else:
        print("❌ PIPELINE FAILED!")
        logger.error(f"❌ Pipeline failed after {duration}")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())