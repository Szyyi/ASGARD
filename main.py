import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import webbrowser

# Add current directory to path to ensure imports work
sys.path.append('.')

# Import project modules
from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineering import FeatureEngineering
from models.terrain_analyser import TerrainAnalyser
from models.movement_predictor import MovementPredictor
from models.evaluation import PredictionEvaluator
from visualisation.visualisation import Visualization
from visualisation.interactive_map import InteractiveMap

def main():
    print("Starting ASGARD predictive tracking system...")
    
    # Load data directly from CSV files
    targets_df = pd.read_csv("data/target_data.csv")
    elevation_df = pd.read_csv("data/elevation_map/elevation_map.csv")
    land_use_df = pd.read_csv("data/land_use/land_use.csv")
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in targets_df.columns:
        targets_df['timestamp'] = pd.to_datetime(targets_df['timestamp'])
    
    # Extract blue forces
    blue_forces_df = targets_df[targets_df['is_blue'] == 1].copy() if 'is_blue' in targets_df.columns else pd.DataFrame()
    
    # Debug column names
    print("Elevation columns:", elevation_df.columns.tolist())
    print("Land use columns:", land_use_df.columns.tolist())
    
    # Fix column names for coordinate compatibility
    if 'latitude' in elevation_df.columns and 'x' not in elevation_df.columns:
        elevation_df.rename(columns={'latitude': 'y', 'longitude': 'x'}, inplace=True)
    
    if 'latitude' in land_use_df.columns and 'x' not in land_use_df.columns:
        land_use_df.rename(columns={'latitude': 'y', 'longitude': 'x'}, inplace=True)
    
    # Fix land use column name
    if 'land_use_type' in land_use_df.columns and 'land_use' not in land_use_df.columns:
        land_use_df['land_use'] = land_use_df.index  # Create a numeric ID
        print("Added 'land_use' column based on index")
    
    print(f"Loaded {len(targets_df)} target observations")
    print(f"Loaded {len(elevation_df)} elevation points")
    print(f"Loaded {len(land_use_df)} land use points")
    print(f"Loaded {len(blue_forces_df)} blue force observations")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    fe = FeatureEngineering(targets_df, elevation_df, land_use_df, verbose=True)
    targets, blue_forces = fe.preprocess_target_data()
    
    # Terrain analysis
    print("\nPerforming terrain analysis...")
    
    # Create copies for TerrainAnalyser with original column names
    elevation_grid_for_terrain = elevation_df.copy()
    land_use_grid_for_terrain = land_use_df.copy()
    
    # Convert back to latitude/longitude for TerrainAnalyser
    if 'x' in elevation_df.columns and 'latitude' not in elevation_df.columns:
        elevation_grid_for_terrain.rename(columns={'x': 'longitude', 'y': 'latitude'}, inplace=True)
    
    if 'x' in land_use_df.columns and 'latitude' not in land_use_df.columns:
        land_use_grid_for_terrain.rename(columns={'x': 'longitude', 'y': 'latitude'}, inplace=True)
    
    terrain_analyser = TerrainAnalyser(
        elevation_grid=elevation_grid_for_terrain,
        land_use_grid=land_use_grid_for_terrain,
        verbose=True
    )
    
    # Create terrain grid and cost surface
    terrain_grid = terrain_analyser.create_terrain_grid()
    vehicle_cost = terrain_analyser.create_cost_surface('vehicle')
    infantry_cost = terrain_analyser.create_cost_surface('infantry')
    
    # Extract grid resolution from terrain grid
    grid_resolution = int(np.sqrt(len(terrain_grid)))
    print(f"Using grid resolution: {grid_resolution}x{grid_resolution}")
    
    # Create mobility network
    mobility_network = terrain_analyser.create_mobility_network('vehicle')
    
    # Create movement predictor
    print("\nInitializing movement predictor...")
    predictor = MovementPredictor(
        targets, 
        blue_forces, 
        vehicle_cost, 
        grid_resolution,
        mobility_network,
        verbose=True
    )
    
    # Generate predictions for all targets
    print("\nGenerating predictions...")
    time_horizons = [15, 30, 60]  # Minutes ahead to predict
    target_ids = targets['target_id'].unique()[:5]  # Limit to first 5 targets for testing
    
    prediction_results = {}
    for horizon in time_horizons:
        print(f"Predicting for {horizon} minute horizon...")
        prediction_results[horizon] = {}
        for target_id in target_ids:
            prediction = predictor.generate_probability_heatmap(
                target_id, 
                horizon, 
                method='integrated'
            )
            if prediction is not None:
                prediction_results[horizon][target_id] = prediction
                print(f"  Generated prediction for target {target_id}")
    
    # Evaluate predictions if we have an evaluation module
    print("\nEvaluating predictions...")
    try:
        # Check for available methods in PredictionEvaluator
        evaluator = PredictionEvaluator(
            targets_df=targets,
            predictions=prediction_results
        )
        
        # Check for available methods
        available_methods = [method for method in dir(evaluator) if not method.startswith('_')]
        print(f"Available evaluation methods: {available_methods}")
        
        # Use available evaluation methods
        if hasattr(evaluator, 'evaluate'):
            evaluation_results = evaluator.evaluate()
            print("Used evaluate() method")
        elif hasattr(evaluator, 'evaluate_predictions'):
            evaluation_results = evaluator.evaluate_predictions()
            print("Used evaluate_predictions() method")
        else:
            print("No suitable evaluation method found")
        
        # Try to print summary if available
        if hasattr(evaluator, 'print_summary'):
            evaluator.print_summary()
        elif hasattr(evaluator, 'summary'):
            print(evaluator.summary())
        elif hasattr(evaluator, 'get_summary'):
            print(evaluator.get_summary())
    except Exception as e:
        print(f"Warning: Could not perform evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize results with static images
    print("\nVisualizing results (static images)...")
    try:
        # Initialize visualization class with proper parameters
        viz = Visualization(
            targets_df=targets, 
            blue_forces_df=blue_forces, 
            terrain_grid_df=terrain_grid,
            verbose=True
        )
        
        # Visualize each prediction
        for horizon in time_horizons:
            for target_id in prediction_results[horizon]:
                pred_data = prediction_results[horizon][target_id]
                # Create visualization
                ax = viz.plot_prediction_heatmap(pred_data)
                # Get the parent figure of the axes
                fig = ax.figure
                # Save the figure
                output_file = os.path.join(output_dir, f"prediction_{target_id}_{horizon}min.png")
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Visualization saved to {output_file}")
    except Exception as e:
        print(f"Warning: Could not generate static visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Generate interactive map visualization
    print("\nGenerating interactive map...")
    try:
        # Prepare predictions in correct format
        formatted_predictions = {}
        for horizon in prediction_results:
            for target_id in prediction_results[horizon]:
                pred_key = f"{target_id}_{horizon}"
                formatted_predictions[pred_key] = prediction_results[horizon][target_id]
        
        # Create interactive map
        interactive_map = InteractiveMap(
            targets_df=targets,
            blue_forces_df=blue_forces,
            terrain_grid_df=terrain_grid,
            predictions=formatted_predictions
        )

        # Generate and save the interactive map
        map_file = os.path.join(output_dir, "interactive_map.html")
        # Force percentage values for width and height
        m = interactive_map.generate_map(output_file=map_file)
        print(f"Interactive map saved to {map_file}")
        
        # Try to open the map in the default browser
        try:
            webbrowser.open('file://' + os.path.abspath(map_file))
            print("Interactive map opened in browser")
        except Exception as e:
            print(f"Warning: Could not open map in browser: {str(e)}")
            print(f"Please open {map_file} manually")
    except Exception as e:
        print(f"Warning: Could not generate interactive map: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Generate terrain report
    print("\nGenerating terrain report...")
    try:
        # Custom implementation for terrain report
        terrain_features = {
            'elevation_range': {
                'min': float(terrain_grid['elevation'].min()) if 'elevation' in terrain_grid.columns else 0,
                'max': float(terrain_grid['elevation'].max()) if 'elevation' in terrain_grid.columns else 0,
                'mean': float(terrain_grid['elevation'].mean()) if 'elevation' in terrain_grid.columns else 0
            },
            'land_use_summary': {},
            'report_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'grid_resolution': grid_resolution,
            'analysis_area': {
                'min_x': float(terrain_grid['x'].min()) if 'x' in terrain_grid.columns else 0,
                'max_x': float(terrain_grid['x'].max()) if 'x' in terrain_grid.columns else 0,
                'min_y': float(terrain_grid['y'].min()) if 'y' in terrain_grid.columns else 0,
                'max_y': float(terrain_grid['y'].max()) if 'y' in terrain_grid.columns else 0
            }
        }
        
        # Try to get land use summary if available
        if 'land_use' in terrain_grid.columns:
            land_use_counts = terrain_grid['land_use'].value_counts().to_dict()
            terrain_features['land_use_summary'] = {str(k): int(v) for k, v in land_use_counts.items()}
        
        # Save as terrain report
        report_file = os.path.join(output_dir, "terrain_report.json")
        
        import json
        with open(report_file, 'w') as f:
            json.dump(terrain_features, f, indent=2, default=str)
        
        print(f"Terrain report saved to {report_file}")
    except Exception as e:
        print(f"Warning: Could not generate terrain report: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nASGARD system execution complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()