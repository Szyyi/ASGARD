# Predictive Target Tracking System

## Military Target Movement Prediction with Multi-modal Terrain-Aware Analysis

## Overview

The Predictive Target Tracking System (PTTS) is an advanced solution designed for battlefield intelligence applications that forecasts enemy target movements based on terrain analysis, historical patterns, and tactical behaviors. This system addresses the critical challenge of maintaining situational awareness when direct observation of targets is not possible.

### Key Capabilities

- **Multi-horizon Prediction**: Generate probability distributions of target locations at 15, 30, 60, and 120-minute intervals
- **Terrain-Aware Analysis**: Incorporates digital elevation models and land use data to model realistic movement constraints
- **Tactical Behavior Modeling**: Accounts for military doctrine, blue force avoidance, and concealment-seeking behaviors
- **Visual Decision Support**: Interactive heatmaps and 3D visualizations for clear tactical understanding
- **Confidence Metrics**: Uncertainty quantification to support risk assessment in decision-making

## Technical Approach

Our solution implements a novel Multi-modal Terrain-Aware Predictive Tracking (MTAPT) algorithm that combines:

1. **Bayesian Movement Analysis**: Learns target movement patterns from historical observations
2. **Terrain Cost Modeling**: Creates mobility cost surfaces based on elevation, slope, and land use
3. **Agent-Based Simulation**: Monte Carlo simulations of target movement considering tactical behaviors
4. **Probabilistic Heatmaps**: Kernel density estimation to visualize probable future locations

## System Architecture


predictive-tracking/
├── data/
│   ├── elevation_map/
│   ├── land_use/
│   └── target_data.csv
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── movement_predictor.py
│   │   ├── terrain_analyzer.py
│   │   └── evaluation.py
│   └── visualization/
│       ├── __init__.py
│       └── interactive_map.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_development.ipynb
├── tests/
│   └── test_models.py
├── requirements.txt
└── README.md


## Data Requirements

The system integrates three primary data sources:

1. **Target Sighting Data** (CSV format)
   - Target ID, class (vehicle, infantry, etc.)
   - Latitude/longitude coordinates
   - Timestamp
   - Optional: Speed, heading

2. **Digital Elevation Model** (CSV or GeoTIFF)
   - Elevation data for the area of operations
   - Coordinate reference system information

3. **Land Use Data** (CSV or GeoTIFF)
   - Classification of terrain (urban, road, forest, open terrain, water)
   - Same spatial coverage as elevation data

## Installation

# Clone the repository
git clone https://github.com/Szyyi/ASGARD.git
cd ASGARD

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## Quick Start

# Example usage
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineering
from src.models.movement_predictor import MovementPredictor
from src.visualization.interactive_map import Visualization

# Load data
loader = DataLoader("./data")
targets_df = loader.load_target_data()
elevation_df = loader.load_elevation_map()
land_use_df = loader.load_land_use()

# Preprocess data
fe = FeatureEngineering(targets_df, elevation_df, land_use_df)
targets, blue_forces = fe.preprocess_target_data()
terrain_grid_df = fe.create_grid_map(resolution=100)[0]
terrain_grid_df = fe.add_terrain_to_grid(terrain_grid_df, fe.create_terrain_cost_map('vehicle'))

# Create prediction model
predictor = MovementPredictor(targets, blue_forces, terrain_grid_df, 100)

# Generate predictions for target #1 at 30 minutes ahead
predictions_30min = predictor.predict_with_terrain(1, 30)
heatmap_30min = predictor.generate_probability_heatmap(1, 30)

# Visualize results
viz = Visualization(targets, blue_forces, terrain_grid_df)
prediction_map = viz.plot_prediction_heatmap(heatmap_30min, 1)
interactive_map = viz.create_interactive_map(1, [heatmap_30min])


## Advanced Features

### Multi-Target Analysis

The system supports tracking multiple targets simultaneously and can analyse patterns of coordination between targets:


# Analyze multiple targets
target_ids = [1, 2, 3]
all_predictions = {}

for target_id in target_ids:
    all_predictions[target_id] = {}
    for horizon in [15, 30, 60]:
        all_predictions[target_id][horizon] = predictor.generate_probability_heatmap(target_id, horizon)


### Tactical Doctrine Integration

Military doctrine parameters can be configured to model specific target behaviors:


# Configure doctrine parameters
Config.BLUE_FORCE_AVOIDANCE = 8000  # Meters to avoid blue forces
Config.TERRAIN_INFLUENCE = 0.8      # Weight of terrain in movement decisions
Config.HISTORICAL_INFLUENCE = 0.2   # Weight of historical patterns


### 3D Terrain Visualization

For comprehensive terrain analysis, the system provides 3D visualization capabilities:


# Create 3D terrain visualization with predicted paths
terrain_3d = viz.create_3d_terrain_plot(target_id=1, predictions=predictions_30min)


## Evaluation Metrics

The system includes built-in evaluation metrics to assess prediction accuracy:

1. **Circular Error Probable (CEP)**: Radius containing 50% of prediction points
2. **Mean Absolute Error (MAE)**: Average distance between predicted and actual positions
3. **Coverage Probability**: Percentage of actual positions falling within prediction heatmap areas

## Decision Support Guidelines

When interpreting prediction results, consider:

1. **Confidence Levels**: Higher density areas represent more likely locations
2. **Terrain Constraints**: Areas with high mobility costs may limit movement options
3. **Time Horizons**: Uncertainty increases with longer prediction windows
4. **Multiple Scenarios**: Consider various tactical behaviors the target might exhibit

## Model Limitations

Important considerations for operational use:

- Predictions assume targets behave rationally according to terrain and tactical constraints
- Sudden changes in target behavior or mission parameters may reduce accuracy
- Urban environments with complex structures have higher uncertainty
- Accuracy decreases with time horizon and sparse historical data

## Future Enhancements

Planned improvements for future versions:

- Real-time data integration from tactical networks
- Deep learning models for pattern recognition in target behaviors
- Multi-target interaction and coordination analysis
- Enhanced visualization for command and control systems

## Project Team

- **Lead Developer**: Szymon Procak & Owen Evason


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to all contributors and the organizers of the Future Forces Defence Hackathon 2025 for the opportunity to develop this solution.

---

*For questions, support, or collaboration opportunities, please contact SzyYP@proton.me