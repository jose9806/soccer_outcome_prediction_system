# Scripts Directory

This directory contains all the executable scripts for the Soccer Outcome Prediction System.

## Available Scripts

### 🚀 Pipeline Scripts

**`run_full_pipeline.py`** - Complete ML pipeline runner
```bash
# Run complete pipeline with default settings
python scripts/run_full_pipeline.py

# Skip data processing, enable optimization
python scripts/run_full_pipeline.py --skip-processing --optimize --trials 20

# Quick training without optimization
python scripts/run_full_pipeline.py --skip-processing --cv-folds 3
```

**`process_complete_dataset.py`** - Data processing and feature engineering
```bash
# Process all data with temporal weighting
python scripts/process_complete_dataset.py

# Process data from specific directory
python scripts/process_complete_dataset.py --data-dir path/to/data
```

**`train_ml_models.py`** - Model training and evaluation
```bash
# Basic training
python scripts/train_ml_models.py

# With hyperparameter optimization
python scripts/train_ml_models.py --optimize --trials 25

# Quick training for testing
python scripts/train_ml_models.py --cv-folds 3
```

### 🔮 Prediction Scripts

**`predict_match.py`** - Main prediction interface
```bash
# Basic prediction
python scripts/predict_match.py "Real Madrid" "Barcelona"

# With specific date and competition
python scripts/predict_match.py "Liverpool" "Manchester City" --date "2025-03-15" --competition "Premier League"

# JSON output for programmatic use
python scripts/predict_match.py "Dep Cali" "Bucaramanga" --json

# Verbose output with debug info
python scripts/predict_match.py "Team A" "Team B" --verbose
```

### 📊 Utility Scripts

**`list_teams.py`** - List and search available teams
```bash
# List all available teams
python scripts/list_teams.py

# Search for teams
python scripts/list_teams.py --search "madrid"

# Just show count
python scripts/list_teams.py --count
```

## How the Prediction System Works

### Input: Just Team Names
The system only requires team names as input:
```bash
python scripts/predict_match.py "Real Madrid" "Barcelona"
```

### Automatic Feature Generation
The system automatically:

1. **Finds Teams**: Matches input names to historical data using flexible matching
2. **Loads History**: Retrieves all historical matches for both teams up to the prediction date
3. **Generates Features**: Creates the same 98+ features used in training:
   - **Temporal Features**: Form, momentum, seasonal patterns
   - **Contextual Features**: Home advantage, rivalry effects, travel fatigue
   - **Advanced Metrics**: xG analysis, efficiency metrics, playing style
   - **Head-to-Head**: Direct matchup history and patterns

4. **Makes Prediction**: Uses trained ML models (XGBoost, LightGBM, RandomForest) in ensemble
5. **Provides Results**: Probabilities, confidence, expected goals, betting recommendations

### Model Architecture
- **XGBoost**: Handles complex feature interactions
- **LightGBM**: Efficient gradient boosting with categorical handling  
- **RandomForest**: Robust ensemble learning
- **Ensemble**: Combines all models for best accuracy

### Feature Engineering Pipeline
The system uses the same feature engineering pipeline for both training and prediction:

```python
# Training: Process historical dataset
features = feature_engine.process_dataset(historical_data)

# Prediction: Generate features for new match
features = feature_engine.generate_match_features(home_team, away_team, date)
```

This ensures consistency between training and prediction phases.

## Common Usage Patterns

### 1. Full Pipeline Setup
```bash
# First time setup - complete pipeline
python scripts/run_full_pipeline.py --optimize

# Later updates - skip processing if data unchanged
python scripts/run_full_pipeline.py --skip-processing
```

### 2. Regular Predictions  
```bash
# Check available teams
python scripts/list_teams.py --search "barcelona"

# Make prediction
python scripts/predict_match.py "FC Barcelona" "Real Madrid"
```

### 3. Batch Predictions
```bash
# JSON output for automation
python scripts/predict_match.py "Team A" "Team B" --json > prediction.json
```

### 4. Development/Testing
```bash
# Quick training for testing changes
python scripts/train_ml_models.py --cv-folds 3

# Verbose prediction output
python scripts/predict_match.py "Team A" "Team B" --verbose
```

## Directory Structure
```
scripts/
├── README.md                    # This file
├── run_full_pipeline.py        # Complete pipeline runner
├── process_complete_dataset.py  # Data processing
├── train_ml_models.py          # Model training  
├── predict_match.py            # Main prediction interface
└── list_teams.py               # Team utilities
```

## Error Handling
All scripts include comprehensive error handling and logging:
- **Input validation**: Checks team names, dates, file paths
- **Graceful degradation**: Works with missing data
- **Informative errors**: Clear error messages and suggestions
- **Logging**: Detailed logs for debugging

## Next Steps
After running the pipeline, you can:
1. Make predictions with `predict_match.py`
2. Integrate predictions into betting systems
3. Set up automated retraining schedules
4. Build web interfaces using the prediction API