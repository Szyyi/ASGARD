import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add current directory to path to ensure imports work
sys.path.append('.')

# Import project modules
from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineering import FeatureEngineering
from models.terrain_analyser import TerrainAnalyser
from models.movement_predictor import MovementPredictor
from models.evaluation import PredictionEvaluator
from visualisation.visualisation import visualize_predictions

def main():
   print("Starting ASGARD predictive tracking system...")
   
   # Initialize data loader
   data_loader = DataLoader(data_folder="data", verbose=True)
   
   # Load data
   targets_df = data_loader.load_target_data()
   elevation_df = data_loader.load_elevation_map()
   land_use_df = data_loader.load_land_use()
   blue_forces_df = data_loader.load_blue_forces()
   
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
   terrain_analyser = TerrainAnalyser(
       elevation_grid=elevation_df,
       land_use_grid=land_use_df,
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
       evaluator = PredictionEvaluator(
           targets_df=targets,
           predictions=prediction_results,
           terrain_grid=terrain_grid
       )
       
       evaluation_results = evaluator.evaluate_all_predictions()
       evaluator.print_evaluation_summary()
   except Exception as e:
       print(f"Warning: Could not perform evaluation: {str(e)}")
   
   # Visualize results
   print("\nVisualizing results...")
   output_dir = "output"
   os.makedirs(output_dir, exist_ok=True)
   
   try:
       for horizon in time_horizons:
           for target_id in prediction_results[horizon]:
               output_file = os.path.join(output_dir, f"prediction_{target_id}_{horizon}min.png")
               visualize_predictions(
                   prediction_results[horizon][target_id],
                   terrain_grid,
                   output_file=output_file
               )
               print(f"  Visualization saved to {output_file}")
   except Exception as e:
       print(f"Warning: Could not generate visualizations: {str(e)}")
   
   # Generate terrain report
   print("\nGenerating terrain report...")
   try:
       terrain_report = terrain_analyser.generate_terrain_report(target_type='vehicle')
       report_file = os.path.join(output_dir, "terrain_report.json")
       
       import json
       with open(report_file, 'w') as f:
           json.dump(terrain_report, f, indent=2, default=str)
       
       print(f"Terrain report saved to {report_file}")
   except Exception as e:
       print(f"Warning: Could not generate terrain report: {str(e)}")
   
   print("\nASGARD system execution complete.")

if __name__ == "__main__":
   try:
       main()
   except Exception as e:
       print(f"Error in main execution: {str(e)}")
       import traceback
       traceback.print_exc()