"""
Movement Predictor Module

This module implements predictive algorithms for forecasting target movements
based on terrain analysis, historical patterns, and tactical behaviors.

Key capabilities:
- Multi-modal prediction incorporating multiple analytical approaches
- Monte Carlo simulation with terrain and doctrine awareness
- Bayesian analysis of historical movement patterns
- Probabilistic heatmap generation for different time horizons
- Confidence scoring for predictions
"""

import numpy as np
import pandas as pd
import networkx as nx # type: ignore
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal, gaussian_kde
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import pickle
import datetime
import os
import math
import time
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MovementPredictor:
    """
    Main class for predicting target movements based on historical data and terrain analysis.
    
    Attributes:
        targets_df (pd.DataFrame): DataFrame containing target observations
        blue_forces_df (pd.DataFrame): DataFrame containing blue force observations
        terrain_grid_df (pd.DataFrame): DataFrame containing terrain grid with costs
        grid_resolution (int): Resolution of the prediction grid
        mobility_network (nx.Graph): NetworkX graph for pathfinding (optional)
        config (dict): Configuration parameters
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, targets_df, blue_forces_df, terrain_grid_df, grid_resolution, 
                 mobility_network=None, config=None, verbose=True):
        """
        Initialize the MovementPredictor.
        
        Args:
            targets_df (pd.DataFrame): DataFrame containing target observations
            blue_forces_df (pd.DataFrame): DataFrame containing blue force observations
            terrain_grid_df (pd.DataFrame): DataFrame containing terrain grid with costs
            grid_resolution (int): Resolution of the prediction grid
            mobility_network (nx.Graph): NetworkX graph for pathfinding (optional)
            config (dict): Configuration parameters (optional)
            verbose (bool): Whether to print detailed information
        """
        self.targets_df = targets_df
        self.blue_forces_df = blue_forces_df
        self.terrain_grid_df = terrain_grid_df
        self.grid_resolution = grid_resolution
        self.mobility_network = mobility_network
        self.verbose = verbose
        
        # Default configuration
        self.default_config = {
            'random_seed': 42,
            'time_horizons': [15, 30, 60, 120],  # Minutes ahead to predict
            'num_simulations': 1000,  # Number of Monte Carlo simulations
            'prediction_methods': ['historical', 'terrain', 'tactical', 'integrated'],
            'terrain_influence': 0.7,  # Weight of terrain in movement prediction (0-1)
            'historical_influence': 0.3,  # Weight of historical patterns (0-1)
            'tactical_influence': 0.5,  # Weight of tactical behavior patterns (0-1)
            'blue_force_avoidance': 5.0,  # Kilometer radius where targets avoid blue forces
            'cache_predictions': True,  # Whether to cache predictions
            'cache_dir': './cache',  # Directory for caching predictions
            'default_speed': {  # Default speeds (km/h) if historical data unavailable
                'vehicle': 20.0,
                'infantry': 3.0,
                'artillery': 10.0,
                'command': 15.0
            },
            'heading_variance': {  # Standard deviation of heading changes (degrees per minute)
                'vehicle': 2.0,
                'infantry': 5.0,
                'artillery': 3.0,
                'command': 2.5
            },
            'concealment_seeking': {  # Propensity to seek concealment (0-1)
                'vehicle': 0.5,
                'infantry': 0.9,
                'artillery': 0.7,
                'command': 0.8
            },
            # Uncertainty growth rate with time (how quickly prediction uncertainty increases)
            'uncertainty_growth': 0.2
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Set random seed
        np.random.seed(self.config['random_seed'])
        
        # Create cache directory if caching is enabled
        if self.config['cache_predictions']:
            os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Create grid map for calculations if needed
        self._prepare_grid_map()
        
        # Store target models
        self.target_models = {}
        
        if self.verbose:
            print("Movement Predictor initialized")
            print(f"Configured for {len(self.config['time_horizons'])} time horizons: {self.config['time_horizons']} minutes")
            print(f"Using prediction methods: {self.config['prediction_methods']}")
    
    def _prepare_grid_map(self):
        """
        Prepare the grid map for calculations.
        
        Creates intermediate data structures for efficient prediction.
        """
        # Check if we have the required data
        if self.terrain_grid_df is None or self.terrain_grid_df.empty:
            if self.verbose:
                print("Warning: No terrain grid data provided")
            return
        
        # Create mappings between grid_id and coordinates
        self.grid_to_coords = {}
        self.coords_to_grid = {}
        
        for idx, row in self.terrain_grid_df.iterrows():
            grid_id = row['grid_id'] if 'grid_id' in row else idx
            coords = (row['latitude'], row['longitude'])
            self.grid_to_coords[grid_id] = coords
            self.coords_to_grid[coords] = grid_id
        
        # Create KD-tree for nearest neighbor searches
        self.grid_tree = cKDTree([(row['latitude'], row['longitude']) for _, row in self.terrain_grid_df.iterrows()])
        
        # Create cost arrays for efficient access
        if 'total_cost' in self.terrain_grid_df.columns:
            # Reshape to 2D grid
            cost_array = self.terrain_grid_df['total_cost'].values.reshape(self.grid_resolution, self.grid_resolution)
            
            # Create interpolator for continuous cost evaluation
            from scipy.interpolate import RectBivariateSpline
            
            # Get coordinate bounds
            lat_min, lat_max = self.terrain_grid_df['latitude'].min(), self.terrain_grid_df['latitude'].max()
            lon_min, lon_max = self.terrain_grid_df['longitude'].min(), self.terrain_grid_df['longitude'].max()
            
            # Create coordinate arrays
            lat_array = np.linspace(lat_min, lat_max, self.grid_resolution)
            lon_array = np.linspace(lon_min, lon_max, self.grid_resolution)
            
            # Create spline interpolator
            self.cost_interpolator = RectBivariateSpline(lat_array, lon_array, cost_array)
            
            if self.verbose:
                print("Created cost interpolator for continuous terrain evaluation")
    
    def historical_movement_analysis(self, target_id):
        """
        Analyze historical movement patterns for a specific target.
        
        Args:
            target_id: Unique identifier for the target
            
        Returns:
            dict: Dictionary of movement statistics and models
        """
        # Check if we have data for this target
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        if target_data.empty:
            if self.verbose:
                print(f"No data found for target ID {target_id}")
            return None
        
        # Sort by timestamp
        target_data = target_data.sort_values('timestamp')
        
        # Basic statistics
        stats = {
            'target_id': target_id,
            'target_class': target_data['target_class'].iloc[0],
            'observations': len(target_data),
            'first_observed': target_data['timestamp'].iloc[0],
            'last_observed': target_data['timestamp'].iloc[-1],
            'avg_speed': target_data['speed'].mean(),
            'max_speed': target_data['speed'].max(),
            'avg_heading': target_data['heading'].mean() % 360,
            'heading_std': target_data['heading'].std(),
            'last_latitude': target_data['latitude'].iloc[-1],
            'last_longitude': target_data['longitude'].iloc[-1],
            'last_speed': target_data['speed'].iloc[-1],
            'last_heading': target_data['heading'].iloc[-1]
        }
        
        # Only build models if we have enough observations
        if len(target_data) >= 3:
            # Features for prediction models
            if all(col in target_data.columns for col in ['hour', 'is_night', 'speed', 'heading']):
                # Prepare features and targets for modeling
                X = target_data[['hour', 'is_night', 'speed', 'heading']].iloc[:-1].values
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Target variables for next step prediction
                y_speed = target_data['speed'].iloc[1:].values
                y_heading = target_data['heading'].iloc[1:].values
                
                # Train random forest models
                # Speed model
                speed_model = GradientBoostingRegressor(
                    n_estimators=50, 
                    learning_rate=0.1, 
                    max_depth=3, 
                    random_state=self.config['random_seed']
                )
                speed_model.fit(X_scaled, y_speed)
                
                # Heading model (needs to handle circular data)
                # Convert angles to sine and cosine components
                y_sin = np.sin(np.radians(y_heading))
                y_cos = np.cos(np.radians(y_heading))
                
                # Train models for sine and cosine components
                heading_sin_model = GradientBoostingRegressor(
                    n_estimators=50, 
                    learning_rate=0.1, 
                    max_depth=3, 
                    random_state=self.config['random_seed']
                )
                heading_sin_model.fit(X_scaled, y_sin)
                
                heading_cos_model = GradientBoostingRegressor(
                    n_estimators=50, 
                    learning_rate=0.1, 
                    max_depth=3, 
                    random_state=self.config['random_seed']
                )
                heading_cos_model.fit(X_scaled, y_cos)
                
                # Store models and scalers
                stats['models'] = {
                    'feature_scaler': scaler,
                    'speed_model': speed_model,
                    'heading_sin_model': heading_sin_model,
                    'heading_cos_model': heading_cos_model
                }
                
                if self.verbose:
                    # Calculate R-squared for speed model
                    speed_r2 = speed_model.score(X_scaled, y_speed)
                    
                    # Calculate approximate R-squared for heading (using sin/cos components)
                    sin_r2 = heading_sin_model.score(X_scaled, y_sin)
                    cos_r2 = heading_cos_model.score(X_scaled, y_cos)
                    heading_r2 = (sin_r2 + cos_r2) / 2  # Simple average
                    
                    print(f"Models for target {target_id}:")
                    print(f"  Speed model R² = {speed_r2:.3f}")
                    print(f"  Heading model R² ≈ {heading_r2:.3f}")
        
        # Add terrain affinity if terrain data is available
        if 'land_use_type' in target_data.columns and 'terrain_cost' in target_data.columns:
            # Calculate how often the target is found in each terrain type
            terrain_counts = target_data['land_use_type'].value_counts(normalize=True)
            
            # Calculate average speed in each terrain type
            terrain_speeds = target_data.groupby('land_use_type')['speed'].mean()
            
            # Calculate terrain preferences
            stats['terrain_affinity'] = terrain_counts.to_dict()
            stats['terrain_speeds'] = terrain_speeds.to_dict()
            
            # Calculate average and standard deviation of terrain costs
            stats['avg_terrain_cost'] = target_data['terrain_cost'].mean()
            stats['std_terrain_cost'] = target_data['terrain_cost'].std()
        
        # Add formation characteristics if available
        if 'formation_dispersion' in target_data.columns:
            stats['avg_formation_dispersion'] = target_data['formation_dispersion'].mean()
        
        # Add tactical behavior patterns
        # Look for patterns such as:
        # - Road following
        # - Forest seeking
        # - Blue force avoidance
        
        if 'dist_to_road' in target_data.columns:
            # Check if target tends to stay near roads
            avg_road_dist = target_data['dist_to_road'].mean()
            stats['road_following'] = 1.0 / (1.0 + avg_road_dist)  # 0-1 scale, higher means more road following
        
        if 'concealment' in target_data.columns:
            # Check if target seeks concealment
            avg_concealment = target_data['concealment'].mean()
            stats['concealment_seeking'] = avg_concealment  # 0-1 scale
        
        if 'blue_proximity' in target_data.columns:
            # Check if target avoids blue forces
            avg_blue_dist = target_data['blue_proximity'].mean()
            if not np.isinf(avg_blue_dist):
                # Calculate avoidance score (higher means more avoidance)
                # Normalize to 0-1 scale, assuming 10km as max relevant distance
                stats['blue_avoidance'] = min(1.0, avg_blue_dist / 10.0)
        
        # Store the analysis results
        self.target_models[target_id] = stats
        
        return stats
    
    def predict_next_position_simple(self, target_id, minutes_ahead):
        """
        Simple prediction based on last known position, heading, and speed.
        
        Args:
            target_id: Unique identifier for the target
            minutes_ahead (int): Minutes in the future to predict
            
        Returns:
            tuple: (latitude, longitude, confidence)
        """
        # Get target data
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        if target_data.empty:
            if self.verbose:
                print(f"No data found for target ID {target_id}")
            return None
        
        # Sort by timestamp and get the last observation
        target_data = target_data.sort_values('timestamp')
        last_obs = target_data.iloc[-1]
        
        # Extract position, speed, and heading
        lat = last_obs['latitude']
        lon = last_obs['longitude']
        
        # Use last known speed or default for target class
        if 'speed' in last_obs and not np.isnan(last_obs['speed']):
            speed = last_obs['speed']
        else:
            target_class = last_obs['target_class']
            speed = self.config['default_speed'].get(target_class, 5.0)
        
        # Use last known heading or random direction
        if 'heading' in last_obs and not np.isnan(last_obs['heading']):
            heading = last_obs['heading']
        else:
            heading = np.random.uniform(0, 360)
        
        # Calculate distance in kilometers
        time_hours = minutes_ahead / 60
        distance_km = speed * time_hours
        
        # Convert heading to radians
        heading_rad = np.radians(heading)
        
        # Calculate new position
        # Using haversine formula approximation
        earth_radius = 6371  # km
        
        # Convert distance to radians
        angular_distance = distance_km / earth_radius
        
        # Calculate destination point
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Calculate new latitude
        new_lat_rad = np.arcsin(
            np.sin(lat_rad) * np.cos(angular_distance) +
            np.cos(lat_rad) * np.sin(angular_distance) * np.cos(heading_rad)
        )
        
        # Calculate new longitude
        new_lon_rad = lon_rad + np.arctan2(
            np.sin(heading_rad) * np.sin(angular_distance) * np.cos(lat_rad),
            np.cos(angular_distance) - np.sin(lat_rad) * np.sin(new_lat_rad)
        )
        
        # Convert back to degrees
        new_lat = np.degrees(new_lat_rad)
        new_lon = np.degrees(new_lon_rad)
        
        # Calculate confidence (decreases with time ahead)
        confidence = 1.0 / (1.0 + self.config['uncertainty_growth'] * minutes_ahead)
        
        return (new_lat, new_lon, confidence)
    
    def predict_with_terrain(self, target_id, minutes_ahead):
        """
        Predict position considering terrain and historical patterns.
        
        Uses Monte Carlo simulation to generate probability distribution.
        
        Args:
            target_id: Unique identifier for the target
            minutes_ahead (int): Minutes in the future to predict
            
        Returns:
            list: List of predicted positions [(lat, lon, probability), ...]
        """
        # Get target analysis or perform it if not available
        if target_id in self.target_models:
            target_stats = self.target_models[target_id]
        else:
            target_stats = self.historical_movement_analysis(target_id)
        
        if target_stats is None:
            if self.verbose:
                print(f"No data available for target ID {target_id}")
            return None
        
        # Get target class
        target_class = target_stats['target_class']
        
        # Get last observed position
        lat = target_stats['last_latitude']
        lon = target_stats['last_longitude']
        
        # Get last speed and heading
        speed = target_stats['last_speed']
        heading = target_stats['last_heading']
        
        # Current time (last observed time)
        last_time = target_stats['last_observed']
        
        # Extract hour and day/night status
        current_hour = last_time.hour
        is_night = 1 if (current_hour >= 18 or current_hour <= 6) else 0
        
        # Create a list to store Monte Carlo simulation results
        num_simulations = self.config['num_simulations']
        simulation_results = []
        
        # Run Monte Carlo simulations
        for sim_idx in range(num_simulations):
            # Start with the last known position
            sim_lat, sim_lon = lat, lon
            sim_speed = speed
            sim_heading = heading
            
            # Simulate movement in small steps
            step_minutes = 5  # 5-minute intervals
            num_steps = max(1, int(minutes_ahead / step_minutes))
            
            # Keep track of the path
            path = [(sim_lat, sim_lon)]
            
            for step in range(num_steps):
                # Use models to predict next speed and heading if available
                if 'models' in target_stats:
                    # Prepare features
                    features = np.array([[current_hour, is_night, sim_speed, sim_heading]])
                    
                    # Standardize
                    X_scaled = target_stats['models']['feature_scaler'].transform(features)
                    
                    # Predict speed
                    predicted_speed = target_stats['models']['speed_model'].predict(X_scaled)[0]
                    
                    # Add random variation to speed
                    speed_std = target_stats.get('speed_std', sim_speed * 0.1)
                    sim_speed = max(0, np.random.normal(predicted_speed, speed_std))
                    
                    # Predict heading components
                    sin_comp = target_stats['models']['heading_sin_model'].predict(X_scaled)[0]
                    cos_comp = target_stats['models']['heading_cos_model'].predict(X_scaled)[0]
                    
                    # Convert back to heading
                    predicted_heading = np.degrees(np.arctan2(sin_comp, cos_comp)) % 360
                    
                    # Add random variation to heading
                    heading_std = self.config['heading_variance'].get(target_class, 3.0) * step_minutes
                    heading_change = np.random.normal(0, heading_std)
                    sim_heading = (predicted_heading + heading_change) % 360
                else:
                    # Simple model: add random variations to current speed and heading
                    speed_change = np.random.normal(0, sim_speed * 0.1)
                    sim_speed = max(0, sim_speed + speed_change)
                    
                    heading_std = self.config['heading_variance'].get(target_class, 3.0) * step_minutes
                    heading_change = np.random.normal(0, heading_std)
                    sim_heading = (sim_heading + heading_change) % 360
                
                # Calculate baseline movement
                # Distance in this step (km)
                step_hours = step_minutes / 60
                distance_km = sim_speed * step_hours
                
                # If we have terrain data, adjust movement based on terrain
                if self.terrain_grid_df is not None and 'total_cost' in self.terrain_grid_df.columns:
                    # Find terrain cost at current position
                    current_cost = self._get_terrain_cost_at_position(sim_lat, sim_lon)
                    
                    # Look at potential next positions
                    candidate_positions = []
                    
                    # Generate potential movement directions
                    # Include the predicted heading and variations
                    heading_options = [
                        sim_heading,  # Main predicted direction
                        (sim_heading + 30) % 360,  # Right turn
                        (sim_heading - 30) % 360,  # Left turn
                        (sim_heading + 60) % 360,  # Sharper right
                        (sim_heading - 60) % 360   # Sharper left
                    ]
                    
                    for h in heading_options:
                        # Calculate potential next position in this direction
                        h_rad = np.radians(h)
                        
                        # Using simple approximation for small distances
                        lat_km_per_degree = 111.32  # km per degree latitude
                        lon_km_per_degree = 111.32 * np.cos(np.radians(sim_lat))  # km per degree longitude
                        
                        lat_change = distance_km * np.cos(h_rad) / lat_km_per_degree
                        lon_change = distance_km * np.sin(h_rad) / lon_km_per_degree
                        
                        new_lat = sim_lat + lat_change
                        new_lon = sim_lon + lon_change
                        
                        # Get terrain cost at new position
                        new_cost = self._get_terrain_cost_at_position(new_lat, new_lon)
                        
                        # Get concealment at new position
                        new_concealment = self._get_concealment_at_position(new_lat, new_lon)
                        
                        # Get blue force influence
                        blue_influence = self._get_blue_influence_at_position(new_lat, new_lon, last_time + datetime.timedelta(minutes=step * step_minutes))
                        
                        # Calculate a weighted score for this position
                        # Lower is better (factors discourage movement)
                        
                        # Terrain factor: higher cost = less desirable
                        terrain_factor = new_cost / max(1.0, current_cost)  # Normalized to current position
                        
                        # Concealment factor: higher concealment = more desirable
                        concealment_factor = 1.0 - new_concealment  # Invert so lower is better
                        
                        # Blue avoidance factor: higher influence = less desirable
                        blue_factor = blue_influence
                        
                        # Heading consistency: preference for continuing in similar direction
                        # Calculate angular difference from original heading
                        heading_diff = min(abs(h - sim_heading), 360 - abs(h - sim_heading))
                        heading_penalty = heading_diff / 180.0  # 0-1 scale
                        
                        # Combine factors with weights from config
                        # Lower score is better
                        score = (
                            terrain_factor * self.config['terrain_influence'] +
                            concealment_factor * self.config['concealment_seeking'].get(target_class, 0.5) +
                            blue_factor * self.config['tactical_influence'] +
                            heading_penalty * 0.2  # Small weight for heading consistency
                        )
                        
                        candidate_positions.append((new_lat, new_lon, score))
                    
                    # Convert scores to probabilities (lower score = higher probability)
                    scores = [pos[2] for pos in candidate_positions]
                    max_score = max(scores)
                    min_score = min(scores)
                    score_range = max_score - min_score
                    
                    if score_range > 0:
                        # Invert and normalize scores to probabilities
                        # Invert so that lower scores become higher probabilities
                        inv_scores = [1 - (score - min_score) / score_range for score in scores]
                        
                        # Normalize to sum to 1
                        total = sum(inv_scores)
                        probs = [s / total for s in inv_scores]
                        
                        # Select position based on probabilities
                        selected_idx = np.random.choice(len(candidate_positions), p=probs)
                        sim_lat, sim_lon = candidate_positions[selected_idx][:2]
                    else:
                        # All scores are equal, pick randomly
                        selected_idx = np.random.choice(len(candidate_positions))
                        sim_lat, sim_lon = candidate_positions[selected_idx][:2]
                else:
                    # No terrain data available, use simple movement model
                    heading_rad = np.radians(sim_heading)
                    
                    # Using simple approximation for small distances
                    lat_km_per_degree = 111.32  # km per degree latitude
                    lon_km_per_degree = 111.32 * np.cos(np.radians(sim_lat))  # km per degree longitude
                    
                    lat_change = distance_km * np.cos(heading_rad) / lat_km_per_degree
                    lon_change = distance_km * np.sin(heading_rad) / lon_km_per_degree
                    
                    sim_lat = sim_lat + lat_change
                    sim_lon = sim_lon + lon_change
                
                # Add current position to path
                path.append((sim_lat, sim_lon))
            
            # Add final position to simulation results
            final_position = path[-1]
            simulation_results.append(final_position)
        
        # Return all simulation results for density calculation
        return simulation_results
    
    def predict_tactical_movement(self, target_id, minutes_ahead):
        """
        Predict movement using tactical behavior analysis.
        
        Incorporates doctrine-based behavior patterns.
        
        Args:
            target_id: Unique identifier for the target
            minutes_ahead (int): Minutes in the future to predict
            
        Returns:
            list: List of predicted positions [(lat, lon, probability), ...]
        """
        # This method extends predict_with_terrain with additional tactical behaviors
        # Get target analysis
        if target_id in self.target_models:
            target_stats = self.target_models[target_id]
        else:
            target_stats = self.historical_movement_analysis(target_id)
        
        if target_stats is None:
            if self.verbose:
                print(f"No data available for target ID {target_id}")
            return None
        
        # Get target class
        target_class = target_stats['target_class']
        
        # Get last observed position
        lat = target_stats['last_latitude']
        lon = target_stats['last_longitude']
        
        # Get last speed and heading
        speed = target_stats['last_speed']
        heading = target_stats['last_heading']
        
        # Current time (last observed time)
        last_time = target_stats['last_observed']
        
        # Extract hour and day/night status
        current_hour = last_time.hour
        is_night = 1 if (current_hour >= 18 or current_hour <= 6) else 0
        
        # Define tactical behavior patterns based on target class
        tactical_patterns = {
            'vehicle': {
                'road_following': 0.8,       # High tendency to follow roads
                'concealment_seeking': 0.4,   # Lower need for concealment
                'heading_consistency': 0.7,   # High directional consistency
                'speed_consistency': 0.8,     # Consistent speed
                'terrain_sensitivity': 0.9    # Very affected by terrain
            },
            'infantry': {
                'road_following': 0.3,       # Low tendency to follow roads
                'concealment_seeking': 0.9,   # High need for concealment
                'heading_consistency': 0.4,   # More winding paths
                'speed_consistency': 0.6,     # Moderately consistent speed
                'terrain_sensitivity': 0.7    # Affected by terrain but can traverse more types
            },
            'artillery': {
                'road_following': 0.7,       # High tendency to follow roads
                'concealment_seeking': 0.8,   # High need for concealment
                'heading_consistency': 0.6,   # Moderate directional consistency
                'speed_consistency': 0.7,     # Moderately consistent speed
                'terrain_sensitivity': 0.8    # Significantly affected by terrain
            },
            'command': {
                'road_following': 0.6,       # Moderate tendency to follow roads
                'concealment_seeking': 0.7,   # Moderate-high need for concealment
                'heading_consistency': 0.6,   # Moderate directional consistency
                'speed_consistency': 0.7,     # Moderately consistent speed
                'terrain_sensitivity': 0.7    # Moderately affected by terrain
            }
        }
        
        # Use observed patterns if available, otherwise use defaults
        behavior_pattern = tactical_patterns.get(target_class, tactical_patterns['infantry'])
        
        # Override with observed behaviors if available
        if 'road_following' in target_stats:
            behavior_pattern['road_following'] = target_stats['road_following']
        
        if 'concealment_seeking' in target_stats:
            behavior_pattern['concealment_seeking'] = target_stats['concealment_seeking']
        
        # Create a list to store Monte Carlo simulation results
        num_simulations = self.config['num_simulations']
        simulation_results = []
        
        # Run Monte Carlo simulations with tactical behaviors
        for sim_idx in range(num_simulations):
            # Start with the last known position
            sim_lat, sim_lon = lat, lon
            sim_speed = speed
            sim_heading = heading
            
            # Simulate movement in small steps
            step_minutes = 5  # 5-minute intervals
            num_steps = max(1, int(minutes_ahead / step_minutes))
            
            # Keep track of the path
            path = [(sim_lat, sim_lon)]
            
            # Simulation state
            current_objective = None
            objective_type = None
            
            # Random chance to select an objective based on target class
            # This simulates purposeful movement toward goals
            if np.random.random() < 0.7:  # 70% chance to have a specific objective
                # Select objective type based on target class
                if target_class == 'vehicle':
                    objective_types = ['road', 'urban', 'destination']
                    weights = [0.5, 0.3, 0.2]
                elif target_class == 'infantry':
                    objective_types = ['concealment', 'high_ground', 'destination']
                    weights = [0.6, 0.3, 0.1]
                elif target_class == 'artillery':
                    objective_types = ['high_ground', 'concealment', 'destination']
                    weights = [0.5, 0.4, 0.1]
                elif target_class == 'command':
                    objective_types = ['urban', 'concealment', 'destination']
                    weights = [0.4, 0.4, 0.2]
                else:
                    objective_types = ['destination', 'concealment', 'road']
                    weights = [0.4, 0.3, 0.3]
                
                # Select objective type
                objective_type = np.random.choice(objective_types, p=weights)
                
                # Select specific objective based on type
                if objective_type == 'road' and 'dist_to_road' in self.terrain_grid_df.columns:
                    # Find road points
                    road_points = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == 'road']
                    if not road_points.empty:
                        # Find nearby road points
                        dists = ((road_points['latitude'] - sim_lat)**2 + 
                                 (road_points['longitude'] - sim_lon)**2)
                        nearest_idx = dists.idxmin()
                        road_point = road_points.loc[nearest_idx]
                        current_objective = (road_point['latitude'], road_point['longitude'])
                
                elif objective_type == 'urban' and 'land_use_type' in self.terrain_grid_df.columns:
                    # Find urban areas
                    urban_points = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == 'urban']
                    if not urban_points.empty:
                        # Find suitable urban points (not too close, not too far)
                        dists = ((urban_points['latitude'] - sim_lat)**2 + 
                                 (urban_points['longitude'] - sim_lon)**2)
                        # Convert to km (approximate)
                        dists_km = np.sqrt(dists) * 111
                        # Filter to reasonable range (1-10 km)
                        in_range = (dists_km > 1) & (dists_km < 10)
                        if in_range.any():
                            candidates = urban_points[in_range]
                            # Randomly select one
                            selected_idx = np.random.choice(candidates.index)
                            urban_point = urban_points.loc[selected_idx]
                            current_objective = (urban_point['latitude'], urban_point['longitude'])
                
                elif objective_type == 'concealment' and 'concealment' in self.terrain_grid_df.columns:
                    # Find areas with good concealment
                    concealed_points = self.terrain_grid_df[self.terrain_grid_df['concealment'] > 0.7]
                    if not concealed_points.empty:
                        # Find suitable concealment points
                        dists = ((concealed_points['latitude'] - sim_lat)**2 + 
                                 (concealed_points['longitude'] - sim_lon)**2)
                        # Convert to km (approximate)
                        dists_km = np.sqrt(dists) * 111
                        # Filter to reasonable range (0.5-5 km)
                        in_range = (dists_km > 0.5) & (dists_km < 5)
                        if in_range.any():
                            candidates = concealed_points[in_range]
                            # Randomly select one
                            selected_idx = np.random.choice(candidates.index)
                            concealed_point = candidates.loc[selected_idx]
                            current_objective = (concealed_point['latitude'], concealed_point['longitude'])
                
                elif objective_type == 'high_ground' and 'elevation' in self.terrain_grid_df.columns:
                    # Find high ground areas
                    high_points = self.terrain_grid_df[self.terrain_grid_df['elevation'] > 
                                                    self.terrain_grid_df['elevation'].quantile(0.8)]
                    if not high_points.empty:
                        # Find suitable high points
                        dists = ((high_points['latitude'] - sim_lat)**2 + 
                                 (high_points['longitude'] - sim_lon)**2)
                        # Convert to km (approximate)
                        dists_km = np.sqrt(dists) * 111
                        # Filter to reasonable range (1-8 km)
                        in_range = (dists_km > 1) & (dists_km < 8)
                        if in_range.any():
                            candidates = high_points[in_range]
                            # Randomly select one
                            selected_idx = np.random.choice(candidates.index)
                            high_point = candidates.loc[selected_idx]
                            current_objective = (high_point['latitude'], high_point['longitude'])
                
                elif objective_type == 'destination':
                    # Random destination in general heading direction
                    # Calculate distance based on speed and time
                    max_distance_km = sim_speed * (minutes_ahead / 60)
                    
                    # Limit to reasonable range
                    distance_km = min(max_distance_km, 10)
                    
                    # Convert to degrees (approximate)
                    distance_deg = distance_km / 111
                    
                    # Random angle within 30 degrees of current heading
                    angle_rad = np.radians((sim_heading + np.random.uniform(-30, 30)) % 360)
                    
                    # Calculate destination
                    dest_lat = sim_lat + distance_deg * np.cos(angle_rad)
                    dest_lon = sim_lon + distance_deg * np.sin(angle_rad)
                    
                    current_objective = (dest_lat, dest_lon)
            
            for step in range(num_steps):
                # Use models to predict next speed and heading if available
                if 'models' in target_stats:
                    # Prepare features
                    features = np.array([[current_hour, is_night, sim_speed, sim_heading]])
                    
                    # Standardize
                    X_scaled = target_stats['models']['feature_scaler'].transform(features)
                    
                    # Predict speed
                    predicted_speed = target_stats['models']['speed_model'].predict(X_scaled)[0]
                    
                    # Add random variation to speed, scaled by speed consistency
                    speed_std = (1 - behavior_pattern['speed_consistency']) * predicted_speed * 0.2
                    sim_speed = max(0, np.random.normal(predicted_speed, speed_std))
                    
                    # Predict heading components
                    sin_comp = target_stats['models']['heading_sin_model'].predict(X_scaled)[0]
                    cos_comp = target_stats['models']['heading_cos_model'].predict(X_scaled)[0]
                    
                    # Convert back to heading
                    predicted_heading = np.degrees(np.arctan2(sin_comp, cos_comp)) % 360
                    
                    # Adjust heading based on objective if one exists
                    if current_objective:
                        # Calculate bearing to objective
                        obj_lat, obj_lon = current_objective
                        bearing = self._calculate_bearing(sim_lat, sim_lon, obj_lat, obj_lon)
                        
                        # Blend current heading with bearing to objective
                        # Higher tactical_influence means more focus on objective
                        objective_weight = self.config['tactical_influence']
                        
                        # Calculate weighted average heading (handling circular values)
                        predicted_heading_rad = np.radians(predicted_heading)
                        bearing_rad = np.radians(bearing)
                        
                        # Convert to Cartesian coordinates
                        x = (1 - objective_weight) * np.cos(predicted_heading_rad) + objective_weight * np.cos(bearing_rad)
                        y = (1 - objective_weight) * np.sin(predicted_heading_rad) + objective_weight * np.sin(bearing_rad)
                        
                        # Convert back to angle
                        predicted_heading = np.degrees(np.arctan2(y, x)) % 360
                    
                    # Add random variation to heading, scaled by heading consistency
                    heading_std = (1 - behavior_pattern['heading_consistency']) * 30
                    heading_change = np.random.normal(0, heading_std)
                    sim_heading = (predicted_heading + heading_change) % 360
                else:
                    # Simple model: add random variations to current speed and heading
                    speed_change = np.random.normal(0, sim_speed * 0.1 * (1 - behavior_pattern['speed_consistency']))
                    sim_speed = max(0, sim_speed + speed_change)
                    
                    # Adjust heading if we have an objective
                    if current_objective:
                        obj_lat, obj_lon = current_objective
                        bearing = self._calculate_bearing(sim_lat, sim_lon, obj_lat, obj_lon)
                        
                        # Blend current heading with bearing to objective
                        objective_weight = self.config['tactical_influence']
                        
                        # Calculate weighted average heading (handling circular values)
                        sim_heading_rad = np.radians(sim_heading)
                        bearing_rad = np.radians(bearing)
                        
                        # Convert to Cartesian coordinates
                        x = (1 - objective_weight) * np.cos(sim_heading_rad) + objective_weight * np.cos(bearing_rad)
                        y = (1 - objective_weight) * np.sin(sim_heading_rad) + objective_weight * np.sin(bearing_rad)
                        
                        # Convert back to angle
                        sim_heading = np.degrees(np.arctan2(y, x)) % 360
                    
                    # Add random variation to heading
                    heading_std = (1 - behavior_pattern['heading_consistency']) * 30
                    heading_change = np.random.normal(0, heading_std)
                    sim_heading = (sim_heading + heading_change) % 360
                
                # Calculate baseline movement distance
                step_hours = step_minutes / 60
                distance_km = sim_speed * step_hours
                
                # If we have terrain data, adjust movement based on terrain and tactical factors
                if self.terrain_grid_df is not None and 'total_cost' in self.terrain_grid_df.columns:
                    # Find terrain cost at current position
                    current_cost = self._get_terrain_cost_at_position(sim_lat, sim_lon)
                    
                    # Look at potential next positions
                    candidate_positions = []
                    
                    # Generate potential movement directions
                    # Include the predicted heading and variations
                    heading_options = [
                        sim_heading,  # Main predicted direction
                        (sim_heading + 30) % 360,  # Right turn
                        (sim_heading - 30) % 360,  # Left turn
                        (sim_heading + 60) % 360,  # Sharper right
                        (sim_heading - 60) % 360   # Sharper left
                    ]
                    
                    for h in heading_options:
                        # Calculate potential next position in this direction
                        h_rad = np.radians(h)
                        
                        # Using simple approximation for small distances
                        lat_km_per_degree = 111.32  # km per degree latitude
                        lon_km_per_degree = 111.32 * np.cos(np.radians(sim_lat))  # km per degree longitude
                        
                        lat_change = distance_km * np.cos(h_rad) / lat_km_per_degree
                        lon_change = distance_km * np.sin(h_rad) / lon_km_per_degree
                        
                        new_lat = sim_lat + lat_change
                        new_lon = sim_lon + lon_change
                        
                        # Get terrain cost at new position
                        new_cost = self._get_terrain_cost_at_position(new_lat, new_lon)
                        
                        # Get concealment at new position
                        new_concealment = self._get_concealment_at_position(new_lat, new_lon)
                        
                        # Get blue force influence
                        blue_influence = self._get_blue_influence_at_position(new_lat, new_lon, last_time + datetime.timedelta(minutes=step * step_minutes))
                        
                        # Get distance to road
                        road_distance = self._get_distance_to_feature(new_lat, new_lon, 'road')
                        
                        # Calculate a weighted score for this position
                        # Lower is better (factors discourage movement)
                        
                        # Terrain factor: higher cost = less desirable
                        terrain_factor = new_cost / max(1.0, current_cost)  # Normalized to current position
                        terrain_factor *= behavior_pattern['terrain_sensitivity']
                        
                        # Concealment factor: higher concealment = more desirable
                        concealment_factor = (1.0 - new_concealment) * behavior_pattern['concealment_seeking']
                        
                        # Blue avoidance factor: higher influence = less desirable
                        blue_factor = blue_influence
                        
                        # Road following factor: closer to road = more desirable for some targets
                        road_factor = 0
                        if not np.isnan(road_distance):
                            # Normalize to 0-1 scale (0 = on road, 1 = far from road)
                            road_factor = min(1.0, road_distance / 2.0)  # Assume 2km is "far"
                            road_factor *= behavior_pattern['road_following']
                        
                        # Heading consistency: preference for continuing in similar direction
                        # Calculate angular difference from original heading
                        heading_diff = min(abs(h - sim_heading), 360 - abs(h - sim_heading))
                        heading_penalty = (heading_diff / 180.0) * behavior_pattern['heading_consistency']
                        
                        # Objective factor: if we have an objective, calculate distance reduction
                        objective_factor = 0
                        if current_objective:
                            obj_lat, obj_lon = current_objective
                            
                            # Calculate current distance to objective
                            current_dist = self._calculate_distance(sim_lat, sim_lon, obj_lat, obj_lon)
                            
                            # Calculate new distance to objective
                            new_dist = self._calculate_distance(new_lat, new_lon, obj_lat, obj_lon)
                            
                            # Calculate if this move gets us closer or farther
                            # Normalize to -1 to 1 range (-1 = getting closer, 1 = getting farther)
                            if current_dist > 0:
                                dist_change = (new_dist - current_dist) / current_dist
                                objective_factor = max(-1, min(1, dist_change))
                                
                                # Scale by tactical influence
                                objective_factor *= self.config['tactical_influence']
                        
                        # Combine factors with weights from config
                        # Lower score is better
                        score = (
                            terrain_factor * self.config['terrain_influence'] +
                            concealment_factor +
                            blue_factor * self.config['tactical_influence'] +
                            road_factor +
                            heading_penalty +
                            objective_factor
                        )
                        
                        candidate_positions.append((new_lat, new_lon, score))
                    
                    # Convert scores to probabilities (lower score = higher probability)
                    scores = [pos[2] for pos in candidate_positions]
                    max_score = max(scores)
                    min_score = min(scores)
                    score_range = max_score - min_score
                    
                    if score_range > 0:
                        # Invert and normalize scores to probabilities
                        # Invert so that lower scores become higher probabilities
                        inv_scores = [1 - (score - min_score) / score_range for score in scores]
                        
                        # Normalize to sum to 1
                        total = sum(inv_scores)
                        probs = [s / total for s in inv_scores]
                        
                        # Select position based on probabilities
                        selected_idx = np.random.choice(len(candidate_positions), p=probs)
                        sim_lat, sim_lon = candidate_positions[selected_idx][:2]
                    else:
                        # All scores are equal, pick randomly
                        selected_idx = np.random.choice(len(candidate_positions))
                        sim_lat, sim_lon = candidate_positions[selected_idx][:2]
                else:
                    # No terrain data available, use simple movement model
                    heading_rad = np.radians(sim_heading)
                    
                    # Using simple approximation for small distances
                    lat_km_per_degree = 111.32  # km per degree latitude
                    lon_km_per_degree = 111.32 * np.cos(np.radians(sim_lat))  # km per degree longitude
                    
                    lat_change = distance_km * np.cos(heading_rad) / lat_km_per_degree
                    lon_change = distance_km * np.sin(heading_rad) / lon_km_per_degree
                    
                    sim_lat = sim_lat + lat_change
                    sim_lon = sim_lon + lon_change
                
                # Add current position to path
                path.append((sim_lat, sim_lon))
                
                # Check if we've reached our objective
                if current_objective:
                    obj_lat, obj_lon = current_objective
                    distance_to_obj = self._calculate_distance(sim_lat, sim_lon, obj_lat, obj_lon)
                    
                    # If within 200m of objective, consider it reached
                    if distance_to_obj < 0.2:
                        # Clear objective or select a new one based on current type
                        if objective_type == 'road' or objective_type == 'urban':
                            # Stay in this area
                            current_objective = None
                        else:
                            # Select a new destination in the general direction
                            # Random angle within 90 degrees of current heading
                            angle_rad = np.radians((sim_heading + np.random.uniform(-45, 45)) % 360)
                            
                            # Distance between 1-5 km
                            distance_km = np.random.uniform(1, 5)
                            distance_deg = distance_km / 111
                            
                            # Calculate new destination
                            dest_lat = sim_lat + distance_deg * np.cos(angle_rad)
                            dest_lon = sim_lon + distance_deg * np.sin(angle_rad)
                            
                            current_objective = (dest_lat, dest_lon)
            
            # Add final position to simulation results
            final_position = path[-1]
            simulation_results.append(final_position)
        
        return simulation_results
    
    def generate_probability_heatmap(self, target_id, minutes_ahead, method='integrated'):
        """
        Generate a probability heatmap for the target's future location.
        
        Args:
            target_id: Unique identifier for the target
            minutes_ahead (int): Minutes in the future to predict
            method (str): Prediction method to use
            
        Returns:
            dict: Heatmap data
        """
        # Check cache first
        cache_key = f"{target_id}_{minutes_ahead}_{method}"
        cache_file = os.path.join(self.config['cache_dir'], f"{cache_key}.pkl")
        
        if self.config['cache_predictions'] and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                
                if self.verbose:
                    print(f"Loaded cached prediction for target {target_id}, {minutes_ahead} minutes ahead")
                
                return cached_result
            except:
                if self.verbose:
                    print(f"Failed to load cached prediction, recalculating")
        
        # Get predictions based on method
        if method == 'simple':
            # Simple prediction (single point)
            prediction = self.predict_next_position_simple(target_id, minutes_ahead)
            if prediction is None:
                return None
            
            predictions = [prediction[:2]]  # Convert to list of (lat, lon) tuples
            confidence = prediction[2]
            
        elif method == 'historical':
            # Historical movement analysis
            predictions = self.predict_with_terrain(target_id, minutes_ahead)
            confidence = 1.0 / (1.0 + 0.1 * minutes_ahead)  # Simple decay with time
            
        elif method == 'tactical':
            # Tactical movement prediction
            predictions = self.predict_tactical_movement(target_id, minutes_ahead)
            confidence = 1.0 / (1.0 + 0.05 * minutes_ahead)  # Slower decay for tactical model
            
        elif method == 'integrated':
            # Integrated approach (combine multiple methods)
            # Get predictions from different methods
            historical_predictions = self.predict_with_terrain(target_id, minutes_ahead)
            tactical_predictions = self.predict_tactical_movement(target_id, minutes_ahead)
            
            # Combine predictions with weights
            if historical_predictions and tactical_predictions:
                # Determine weights based on time horizon
                # For longer horizons, tactical model gets more weight
                tactical_weight = min(0.8, 0.4 + 0.01 * minutes_ahead)
                historical_weight = 1.0 - tactical_weight
                
                # Sample from both prediction sets
                num_historical = int(self.config['num_simulations'] * historical_weight)
                num_tactical = self.config['num_simulations'] - num_historical
                
                # Ensure we have enough predictions
                if len(historical_predictions) < num_historical:
                    num_historical = len(historical_predictions)
                    num_tactical = self.config['num_simulations'] - num_historical
                
                if len(tactical_predictions) < num_tactical:
                    num_tactical = len(tactical_predictions)
                    num_historical = self.config['num_simulations'] - num_tactical
                
                # Combine predictions
                predictions = (
                    historical_predictions[:num_historical] +
                    tactical_predictions[:num_tactical]
                )
                
                # Higher confidence for integrated approach
                confidence = 1.0 / (1.0 + 0.03 * minutes_ahead)
            elif historical_predictions:
                predictions = historical_predictions
                confidence = 1.0 / (1.0 + 0.1 * minutes_ahead)
            elif tactical_predictions:
                predictions = tactical_predictions
                confidence = 1.0 / (1.0 + 0.05 * minutes_ahead)
            else:
                return None
        else:
            if self.verbose:
                print(f"Unknown prediction method: {method}")
            return None
        
        if predictions is None or len(predictions) == 0:
            return None
        
        # Convert predictions to arrays
        pred_lats = np.array([p[0] for p in predictions])
        pred_lons = np.array([p[1] for p in predictions])
        
        # Create a grid for the heatmap
        # Determine bounds with some padding
        lat_range = np.max(pred_lats) - np.min(pred_lats)
        lon_range = np.max(pred_lons) - np.min(pred_lons)
        
        # Add padding (10% on each side)
        padding_factor = 0.1
        lat_min = np.min(pred_lats) - padding_factor * lat_range
        lat_max = np.max(pred_lats) + padding_factor * lat_range
        lon_min = np.min(pred_lons) - padding_factor * lon_range
        lon_max = np.max(pred_lons) + padding_factor * lon_range
        
        # Create grid
        grid_size = 100  # Resolution of the heatmap
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Use kernel density estimation to create the heatmap
        positions = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()]).T
        values = np.vstack([pred_lats, pred_lons]).T
        
        # Calculate optimal bandwidth
        bandwidth = 0.001 * np.sqrt(minutes_ahead / 15.0)  # Scale with prediction horizon
        
        # Create kernel density estimator
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(values)
        
        # Get density values
        log_density = kde.score_samples(positions)
        density = np.exp(log_density).reshape(lat_mesh.shape)
        
        # Normalize to 0-1 range
        if density.max() > 0:
            density = density / density.max()
        
        # Calculate confidence regions (68%, 90%, 95%)
        confidence_regions = self._calculate_confidence_regions(density, lat_grid, lon_grid)
        
        # Create result dictionary
        result = {
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'density': density,
            'predictions': predictions,
            'confidence': confidence,
            'minutes_ahead': minutes_ahead,
            'method': method,
            'confidence_regions': confidence_regions,
            'target_id': target_id
        }
        
        # Get target information
        if target_id in self.target_models:
            result['target_class'] = self.target_models[target_id]['target_class']
            result['last_latitude'] = self.target_models[target_id]['last_latitude']
            result['last_longitude'] = self.target_models[target_id]['last_longitude']
            result['last_heading'] = self.target_models[target_id]['last_heading']
            result['last_speed'] = self.target_models[target_id]['last_speed']
        
        # Cache the result
        if self.config['cache_predictions']:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                if self.verbose:
                    print(f"Failed to cache prediction")
        
        return result
    
    def _calculate_confidence_regions(self, density, lat_grid, lon_grid):
        """
        Calculate confidence regions from the probability density.
        
        Args:
            density (numpy.ndarray): 2D array of probability density values
            lat_grid (numpy.ndarray): Array of latitude values
            lon_grid (numpy.ndarray): Array of longitude values
            
        Returns:
            dict: Confidence regions for different probability levels
        """
        # Flatten arrays
        flat_density = density.flatten()
        flat_lats = np.repeat(lat_grid, len(lon_grid))
        flat_lons = np.tile(lon_grid, len(lat_grid))
        
        # Sort points by density in descending order
        sorted_indices = np.argsort(flat_density)[::-1]
        sorted_density = flat_density[sorted_indices]
        sorted_lats = flat_lats[sorted_indices]
        sorted_lons = flat_lons[sorted_indices]
        
        # Calculate cumulative density
        cumulative_density = np.cumsum(sorted_density) / np.sum(sorted_density)
        
        # Define confidence levels
        confidence_levels = {
            68: 0.68,  # 1 sigma
            90: 0.90,  # ~1.6 sigma
            95: 0.95   # ~2 sigma
        }
        
        confidence_regions = {}
        
        for level, prob in confidence_levels.items():
            # Find the threshold that gives this confidence level
            idx = np.searchsorted(cumulative_density, prob)
            if idx >= len(sorted_lats):
                idx = len(sorted_lats) - 1
            
            # Get points within this confidence region
            region_lats = sorted_lats[:idx+1]
            region_lons = sorted_lons[:idx+1]
            
            # Calculate area (in km²)
            # Use convex hull to estimate area
            from scipy.spatial import ConvexHull
            try:
                points = np.vstack((region_lons, region_lats)).T
                hull = ConvexHull(points)
                
                # Area in square degrees
                area_sq_deg = hull.volume
                
                # Convert to km² (approximate at this latitude)
                lat_center = np.mean(region_lats)
                lon_km_per_deg = 111.32 * np.cos(np.radians(lat_center))
                lat_km_per_deg = 111.32
                
                area_km2 = area_sq_deg * lon_km_per_deg * lat_km_per_deg
            except:
                # Default value if calculation fails
                area_km2 = 0
            
            # Store the region
            confidence_regions[level] = {
                'points': list(zip(region_lats, region_lons)),
                'area_km2': area_km2,
                'density_threshold': sorted_density[idx]
            }
        
        return confidence_regions
    
    def _get_terrain_cost_at_position(self, lat, lon):
        """
        Get terrain cost at a specific position.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            float: Terrain cost (higher is more difficult)
        """
        # Check if we have terrain data
        if self.terrain_grid_df is None or 'total_cost' not in self.terrain_grid_df.columns:
            return 1.0  # Default cost
        
        # Use interpolation if available
        if hasattr(self, 'cost_interpolator'):
            # Check if the point is within the bounds
            lat_min, lat_max = self.terrain_grid_df['latitude'].min(), self.terrain_grid_df['latitude'].max()
            lon_min, lon_max = self.terrain_grid_df['longitude'].min(), self.terrain_grid_df['longitude'].max()
            
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                try:
                    # Use bivariate spline interpolation
                    cost = float(self.cost_interpolator(lat, lon))
                    return max(0.1, cost)
                except:
                    # Fallback to nearest neighbor
                    pass
        
        # Find nearest grid point
        distances, indices = self.grid_tree.query([(lat, lon)], k=1)
        nearest_idx = indices[0]
        
        # Get cost from nearest point
        cost = self.terrain_grid_df.iloc[nearest_idx]['total_cost']
        
        return max(0.1, cost)
    
    def _get_concealment_at_position(self, lat, lon):
        """
        Get concealment value at a specific position.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            float: Concealment value (0-1, higher is better concealed)
        """
        # Check if we have concealment data
        if self.terrain_grid_df is None or 'concealment' not in self.terrain_grid_df.columns:
            return 0.0  # Default value
        
        # Find nearest grid point
        distances, indices = self.grid_tree.query([(lat, lon)], k=1)
        nearest_idx = indices[0]
        
        # Get concealment from nearest point
        concealment = self.terrain_grid_df.iloc[nearest_idx]['concealment']
        
        return concealment
    
    def _get_blue_influence_at_position(self, lat, lon, timestamp):
        """
        Calculate the influence of blue forces at a specific position and time.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            timestamp: Timestamp for blue force positions
            
        Returns:
            float: Blue force influence (0-1, higher means stronger influence)
        """
        if self.blue_forces_df is None or self.blue_forces_df.empty:
            return 0.0
        
        # Get blue forces at or before the given timestamp
        try:
            timestamp_pd = pd.to_datetime(timestamp)
            recent_blue = self.blue_forces_df[self.blue_forces_df['timestamp'] <= timestamp_pd].copy()
        except:
            # If timestamp conversion fails, use all blue forces
            recent_blue = self.blue_forces_df.copy()
        
        if recent_blue.empty:
            return 0.0
        
        # Get most recent position for each blue force
        recent_blue = recent_blue.sort_values('timestamp').groupby('blue_id').last().reset_index()
        
        # Calculate distance to each blue force (in km)
        R = 6371  # Earth radius in kilometers
        dlat = np.radians(recent_blue['latitude'] - lat)
        dlon = np.radians(recent_blue['longitude'] - lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(recent_blue['latitude'])) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = R * c
        
        # Calculate influence based on distance and blue force avoidance radius
        max_influence_dist = self.config['blue_force_avoidance']  # km
        influences = 1.0 / (1.0 + (distances / max_influence_dist)**2)
        
        # Return maximum influence from any blue force
        return influences.max()
    
    def _get_distance_to_feature(self, lat, lon, feature_type):
        """
        Calculate distance to the nearest terrain feature of a specific type.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            feature_type (str): Type of feature ('road', 'urban', 'forest', etc.)
            
        Returns:
            float: Distance in kilometers (or np.nan if feature not found)
        """
        if self.terrain_grid_df is None or 'land_use_type' not in self.terrain_grid_df.columns:
            return np.nan
        
        # Find points with the specified feature type
        feature_points = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == feature_type]
        
        if feature_points.empty:
            return np.nan
        
        # Create a KD-tree for the feature points
        tree = cKDTree(feature_points[['latitude', 'longitude']].values)
        
        # Find distance to nearest feature point
        distance, _ = tree.query([lat, lon], k=1)
        
        # Convert to kilometers (approximate)
        distance_km = distance * 111  # Very rough conversion
        
        return distance_km
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing from point 1 to point 2.
        
        Args:
            lat1, lon1: Coordinates of point 1
            lat2, lon2: Coordinates of point 2
            
        Returns:
            float: Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Calculate bearing
        y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
        bearing_rad = np.arctan2(y, x)
        
        # Convert to degrees
        bearing_deg = np.degrees(bearing_rad)
        
        # Normalize to 0-360
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points.
        
        Args:
            lat1, lon1: Coordinates of point 1
            lat2, lon2: Coordinates of point 2
            
        Returns:
            float: Distance in kilometers
        """
        # Haversine formula
        R = 6371  # Earth radius in kilometers
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def batch_predict_targets(self, time_horizon=30, method='integrated'):
        """
        Generate predictions for all targets at a specific time horizon.
        
        Args:
            time_horizon (int): Minutes in the future to predict
            method (str): Prediction method to use
            
        Returns:
            dict: Dictionary of prediction results by target_id
        """
        if self.targets_df is None or self.targets_df.empty:
            if self.verbose:
                print("No target data available")
            return {}
        
        # Get unique target IDs
        target_ids = self.targets_df['target_id'].unique()
        
        # Generate predictions for each target
        results = {}
        
        for target_id in target_ids:
            if self.verbose:
                print(f"Predicting target {target_id} at {time_horizon} minutes ahead")
            
            # Skip blue forces
            target_data = self.targets_df[self.targets_df['target_id'] == target_id]
            if 'is_blue' in target_data.columns and target_data['is_blue'].iloc[0] == 1:
                continue
            
            # Generate prediction
            prediction = self.generate_probability_heatmap(target_id, time_horizon, method)
            
            if prediction is not None:
                results[target_id] = prediction
        
        return results
    
    def evaluate_prediction_accuracy(self, target_id, prediction_time, actual_time, method='integrated'):
        """
        Evaluate the accuracy of a prediction against actual observations.
        
        Args:
            target_id: Unique identifier for the target
            prediction_time: Timestamp to predict from
            actual_time: Timestamp of actual observation to compare against
            method (str): Prediction method to use
            
        Returns:
            dict: Accuracy metrics
        """
        # Get target data
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        if target_data.empty:
            if self.verbose:
                print(f"No data found for target ID {target_id}")
            return None
        
        # Ensure timestamps are datetime
        prediction_time = pd.to_datetime(prediction_time)
        actual_time = pd.to_datetime(actual_time)
        
        # Get time difference in minutes
        time_diff = (actual_time - prediction_time).total_seconds() / 60
        
        if time_diff <= 0:
            if self.verbose:
                print(f"Actual time must be after prediction time")
            return None
        
        # Get prediction and actual positions
        prediction_data = target_data[target_data['timestamp'] == prediction_time]
        actual_data = target_data[target_data['timestamp'] == actual_time]
        
        if prediction_data.empty or actual_data.empty:
            if self.verbose:
                print(f"Missing observation at prediction or actual time")
            return None
        
        # Get coordinates
        pred_lat = prediction_data['latitude'].iloc[0]
        pred_lon = prediction_data['longitude'].iloc[0]
        actual_lat = actual_data['latitude'].iloc[0]
        actual_lon = actual_data['longitude'].iloc[0]
        
        # Generate prediction
        prediction_result = self.generate_probability_heatmap(target_id, time_diff, method)
        
        if prediction_result is None:
            if self.verbose:
                print(f"Failed to generate prediction")
            return None
        
        # Calculate distance between predicted and actual position
        # For point prediction, use the highest probability point
        density = prediction_result['density']
        lat_grid = prediction_result['lat_grid']
        lon_grid = prediction_result['lon_grid']
        
        # Find maximum density point
        max_idx = np.argmax(density)
        max_i, max_j = np.unravel_index(max_idx, density.shape)
        max_lat = lat_grid[max_i]
        max_lon = lon_grid[max_j]
        
        # Calculate error distance
        error_distance = self._calculate_distance(max_lat, max_lon, actual_lat, actual_lon)
        
        # Calculate if actual position is within confidence regions
        in_regions = {}
        for level, region in prediction_result['confidence_regions'].items():
            threshold = region['density_threshold']
            # Get density at actual position
            actual_density = self._interpolate_density(actual_lat, actual_lon, lat_grid, lon_grid, density)
            in_regions[level] = actual_density >= threshold
        
        # Calculate CEP (Circular Error Probable) - radius containing 50% of predictions
        predictions = prediction_result['predictions']
        distances = [self._calculate_distance(p[0], p[1], actual_lat, actual_lon) for p in predictions]
        cep_50 = np.percentile(distances, 50)
        cep_90 = np.percentile(distances, 90)
        
        # Calculate percentage of predictions closer than actual position
        closer_percent = sum(1 for d in distances if d < error_distance) / len(distances) * 100
        
        # Calculate log likelihood of actual position
        log_likelihood = np.log(max(1e-10, actual_density))
        
        # Return metrics
        metrics = {
            'error_distance_km': error_distance,
            'prediction_lat': max_lat,
            'prediction_lon': max_lon,
            'actual_lat': actual_lat,
            'actual_lon': actual_lon,
            'time_horizon_minutes': time_diff,
            'in_confidence_regions': in_regions,
            'cep_50': cep_50,
            'cep_90': cep_90,
            'closer_percent': closer_percent,
            'log_likelihood': log_likelihood,
            'method': method
        }
        
        if self.verbose:
            print(f"Prediction error: {error_distance:.2f} km")
            print(f"In confidence regions: {in_regions}")
        
        return metrics
    
    def _interpolate_density(self, lat, lon, lat_grid, lon_grid, density):
        """
        Interpolate density value at a specific position.
        
        Args:
            lat, lon: Position coordinates
            lat_grid, lon_grid: Grid coordinates
            density: Density values
            
        Returns:
            float: Interpolated density value
        """
        # Check if point is within grid bounds
        if lat < lat_grid.min() or lat > lat_grid.max() or lon < lon_grid.min() or lon > lon_grid.max():
            return 0.0
        
        # Find nearest grid indices
        i = np.searchsorted(lat_grid, lat) - 1
        j = np.searchsorted(lon_grid, lon) - 1
        
        # Ensure indices are within bounds
        i = max(0, min(i, len(lat_grid) - 2))
        j = max(0, min(j, len(lon_grid) - 2))
        
        # Get surrounding grid points
        lat1, lat2 = lat_grid[i], lat_grid[i+1]
        lon1, lon2 = lon_grid[j], lon_grid[j+1]
        
        # Get density values at surrounding points
        q11 = density[i, j]
        q12 = density[i, j+1]
        q21 = density[i+1, j]
        q22 = density[i+1, j+1]
        if len(density.shape) == 2 and min(i+1, j+1) < max(density.shape):
            q11 = density[i, j]
            if j+1 < density.shape[1]:
                q12 = density[i, j+1]
            if i+1 < density.shape[0]:
                q21 = density[i+1, j]
            if i+1 < density.shape[0] and j+1 < density.shape[1]:
                q22 = density[i+1, j+1]
        
        # Bilinear interpolation
        lat_ratio = (lat - lat1) / (lat2 - lat1) if lat2 > lat1 else 0
        lon_ratio = (lon - lon1) / (lon2 - lon1) if lon2 > lon1 else 0
        
        interpolated = (
            q11 * (1 - lat_ratio) * (1 - lon_ratio) +
            q12 * (1 - lat_ratio) * lon_ratio +
            q21 * lat_ratio * (1 - lon_ratio) +
            q22 * lat_ratio * lon_ratio
        )
        
        return interpolated
    
    def predict_evasive_maneuvers(self, target_id, blue_detection_time, minutes_ahead=30):
        """
        Predict evasive maneuvers after blue force detection.
        
        Args:
            target_id: Unique identifier for the target
            blue_detection_time: Timestamp when target is detected by blue forces
            minutes_ahead (int): Minutes to predict after detection
            
        Returns:
            dict: Prediction results
        """
        # This method simulates how a target might react when it knows it's been detected
        
        # Get target analysis
        if target_id in self.target_models:
            target_stats = self.target_models[target_id]
        else:
            target_stats = self.historical_movement_analysis(target_id)
        
        if target_stats is None:
            if self.verbose:
                print(f"No data available for target ID {target_id}")
            return None
        
        # Get target class and last observed position
        target_class = target_stats['target_class']
        lat = target_stats['last_latitude']
        lon = target_stats['last_longitude']
        speed = target_stats['last_speed']
        heading = target_stats['last_heading']
        
        # Define evasive behavior patterns based on target class
        evasive_patterns = {
            'vehicle': {
                'speed_increase': 1.5,      # Increase speed by 50%
                'concealment_seeking': 0.8,  # High priority for concealment
                'direction_change': 90,      # Significant direction change
                'dispersion': True           # Split into multiple units
            },
            'infantry': {
                'speed_increase': 1.2,      # Increase speed by 20%
                'concealment_seeking': 0.9,  # Very high priority for concealment
                'direction_change': 120,     # Major direction change
                'dispersion': True           # Split into multiple units
            },
            'artillery': {
                'speed_increase': 1.3,      # Increase speed by 30%
                'concealment_seeking': 0.8,  # High priority for concealment
                'direction_change': 60,      # Moderate direction change
                'dispersion': False          # Maintain formation
            },
            'command': {
                'speed_increase': 1.4,      # Increase speed by 40%
                'concealment_seeking': 0.9,  # Very high priority for concealment
                'direction_change': 90,      # Significant direction change
                'dispersion': False          # Maintain formation
            }
        }
        
        # Get evasive pattern for this target class
        evasive_pattern = evasive_patterns.get(target_class, evasive_patterns['infantry'])
        
        # Modify config for evasive prediction
        evasive_config = self.config.copy()
        evasive_config['concealment_seeking'] = {
            k: max(v, evasive_pattern['concealment_seeking']) 
            for k, v in self.config['concealment_seeking'].items()
        }
        evasive_config['tactical_influence'] = 0.8  # Higher tactical influence during evasion
        
        # Create a temporary predictor with evasive config
        evasive_predictor = MovementPredictor(
            self.targets_df,
            self.blue_forces_df,
            self.terrain_grid_df,
            self.grid_resolution,
            self.mobility_network,
            evasive_config,
            verbose=False
        )
        
        # Copy target models
        evasive_predictor.target_models = self.target_models.copy()
        
        # Modify target model for evasion
        if target_id in evasive_predictor.target_models:
            # Increase speed
            evasive_predictor.target_models[target_id]['last_speed'] *= evasive_pattern['speed_increase']
            
            # Change direction by specified amount (randomly left or right)
            direction_change = evasive_pattern['direction_change'] * (1 if np.random.random() < 0.5 else -1)
            evasive_predictor.target_models[target_id]['last_heading'] = (heading + direction_change) % 360
        
        # Generate prediction with evasive behavior
        evasive_prediction = evasive_predictor.generate_probability_heatmap(target_id, minutes_ahead, 'tactical')
        
        # Create dispersed predictions if applicable
        if evasive_pattern['dispersion']:
            # Create multiple dispersed units
            num_dispersed = 3  # Number of dispersed units
            dispersed_predictions = []
            
            for i in range(num_dispersed):
                # Create a copy of the target model with variations
                if target_id in evasive_predictor.target_models:
                    # Random direction change (within +/- 30 degrees of evasive direction)
                    disperse_direction = np.random.uniform(-30, 30)
                    evasive_predictor.target_models[target_id]['last_heading'] = (heading + direction_change + disperse_direction) % 360
                    
                    # Random speed variation
                    speed_factor = np.random.uniform(0.8, 1.2)
                    evasive_predictor.target_models[target_id]['last_speed'] = speed * evasive_pattern['speed_increase'] * speed_factor
                
                # Generate prediction for this dispersed unit
                dispersed_prediction = evasive_predictor.generate_probability_heatmap(target_id, minutes_ahead, 'tactical')
                
                if dispersed_prediction is not None:
                    dispersed_predictions.append(dispersed_prediction)
            
            # Combine dispersed predictions if available
            if dispersed_predictions:
                # Combine density grids
                combined_density = np.zeros_like(evasive_prediction['density'])
                for pred in dispersed_predictions:
                    combined_density += pred['density']
                
                # Normalize
                if combined_density.max() > 0:
                    combined_density /= combined_density.max()
                
                # Update evasive prediction with combined density
                evasive_prediction['density'] = combined_density
                evasive_prediction['dispersed'] = True
        
        return evasive_prediction
    
    def predict_group_coordination(self, target_ids, minutes_ahead=30, method='integrated'):
        """
        Predict coordinated movement of a group of targets.
        
        Args:
            target_ids: List of target IDs in the group
            minutes_ahead (int): Minutes in the future to predict
            method (str): Prediction method to use
            
        Returns:
            dict: Prediction results for the group
        """
        if not target_ids:
            if self.verbose:
                print("No target IDs provided")
            return None
        
        # Get individual predictions
        individual_predictions = {}
        for target_id in target_ids:
            prediction = self.generate_probability_heatmap(target_id, minutes_ahead, method)
            if prediction is not None:
                individual_predictions[target_id] = prediction
        
        if not individual_predictions:
            if self.verbose:
                print("No valid predictions generated")
            return None
        
        # Determine if targets form a group
        # Check if they are of the same class
        target_classes = [self.target_models[target_id]['target_class'] 
                        for target_id in target_ids 
                        if target_id in self.target_models]
        
        has_common_class = len(set(target_classes)) == 1 if target_classes else False
        
        # Check if they are in proximity
        last_positions = [
            (self.target_models[target_id]['last_latitude'], self.target_models[target_id]['last_longitude'])
            for target_id in target_ids
            if target_id in self.target_models
        ]
        
        # Calculate pairwise distances
        proximity = False
        if len(last_positions) >= 2:
            distances = []
            for i in range(len(last_positions)):
                for j in range(i+1, len(last_positions)):
                    lat1, lon1 = last_positions[i]
                    lat2, lon2 = last_positions[j]
                    dist = self._calculate_distance(lat1, lon1, lat2, lon2)
                    distances.append(dist)
            
            # Check if maximum distance is less than 2km
            proximity = max(distances) < 2.0 if distances else False
        
        # If targets form a group, adjust predictions for coordination
        if has_common_class and proximity:
            # Get common target class
            common_class = target_classes[0]
            
            # Create combined prediction
            # Get first prediction to use as template
            template_pred = next(iter(individual_predictions.values()))
            
            # Create a combined density grid
            combined_density = np.zeros_like(template_pred['density'])
            
            # Combine using weighted average
            for target_id, prediction in individual_predictions.items():
                weight = 1.0 / len(individual_predictions)  # Equal weights by default
                combined_density += prediction['density'] * weight
            
            # Apply coordination pattern based on target class
            if common_class == 'vehicle':
                # Vehicles tend to move in columns on roads
                # Narrow the density along the primary axis
                combined_density = self._narrow_density_along_axis(combined_density, 0.7)
            elif common_class == 'infantry':
                # Infantry tends to spread out more
                # Add dispersion
                combined_density = gaussian_filter(combined_density, sigma=1.0)
            elif common_class == 'artillery':
                # Artillery tends to cluster in defensive positions
                # Concentrate the density
                combined_density = combined_density ** 1.5
                if combined_density.max() > 0:
                    combined_density /= combined_density.max()
            
            # Create a new prediction result based on the template
            coordinated_prediction = template_pred.copy()
            coordinated_prediction['density'] = combined_density
            coordinated_prediction['coordinated'] = True
            coordinated_prediction['target_ids'] = target_ids
            coordinated_prediction['target_class'] = common_class
            
            return coordinated_prediction
        else:
            # If not a coordinated group, return individual predictions
            return individual_predictions
    
    def _narrow_density_along_axis(self, density, factor=0.7):
        """
        Narrow the density distribution along its principal axis.
        
        Args:
            density (numpy.ndarray): 2D density grid
            factor (float): Factor to narrow by (0-1)
            
        Returns:
            numpy.ndarray: Narrowed density grid
        """
        # Find the principal axis using PCA
        from sklearn.decomposition import PCA
        
        # Get non-zero density points
        nonzero = density > 0.1 * density.max()
        if not np.any(nonzero):
            return density
        
        y_indices, x_indices = np.where(nonzero)
        points = np.column_stack([y_indices, x_indices])
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # Get principal components and origin (mean)
        components = pca.components_
        origin = pca.mean_
        
        # Create a new grid
        result = np.zeros_like(density)
        
        # For each point in the output grid
        for i in range(density.shape[0]):
            for j in range(density.shape[1]):
                # Convert to the PCA space
                point = np.array([i, j])
                centered = point - origin
                coordinates = np.dot(centered, components.T)
                
                # Scale the minor axis (second component)
                coordinates[1] *= factor
                
                # Convert back to original space
                new_point = np.dot(coordinates, components) + origin
                new_i, new_j = int(round(new_point[0])), int(round(new_point[1]))
                
                # Check if within bounds
                if 0 <= new_i < density.shape[0] and 0 <= new_j < density.shape[1]:
                    result[i, j] = density[new_i, new_j]
        
        return result

    def generate_multi_horizon_prediction(self, target_id, time_horizons=None, method='integrated'):
        """
        Generate predictions across multiple time horizons.
        
        Args:
            target_id: Unique identifier for the target
            time_horizons (list): List of time horizons in minutes
            method (str): Prediction method to use
            
        Returns:
            dict: Dictionary of predictions by time horizon
        """
        if time_horizons is None:
            time_horizons = self.config['time_horizons']
        
        results = {}
        
        for horizon in time_horizons:
            prediction = self.generate_probability_heatmap(target_id, horizon, method)
            if prediction is not None:
                results[horizon] = prediction
        
        return results
    
    def save_models(self, output_file="target_models.pkl"):
        """
        Save target models to a file.
        
        Args:
            output_file (str): Path to output file
            
        Returns:
            bool: Success indicator
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save target models (excluding sklearn models that may not pickle well)
            saveable_models = {}
            
            for target_id, model in self.target_models.items():
                saveable_model = model.copy()
                
                # Remove complex models that might not pickle well
                if 'models' in saveable_model:
                    del saveable_model['models']
                
                saveable_models[target_id] = saveable_model
            
            with open(output_file, 'wb') as f:
                pickle.dump(saveable_models, f)
            
            if self.verbose:
                print(f"Saved target models to {output_file}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, input_file="target_models.pkl"):
        """
        Load target models from a file.
        
        Args:
            input_file (str): Path to input file
            
        Returns:
            bool: Success indicator
        """
        try:
            with open(input_file, 'rb') as f:
                loaded_models = pickle.load(f)
            
            # Merge loaded models with existing ones
            self.target_models.update(loaded_models)
            
            if self.verbose:
                print(f"Loaded {len(loaded_models)} target models from {input_file}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading models: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineering
    
    # Load data
    loader = DataLoader(data_folder="./data", verbose=True)
    targets_df = loader.load_target_data()
    elevation_df = loader.load_elevation_map()
    land_use_df = loader.load_land_use()
    
    # Create feature engineering
    fe = FeatureEngineering(targets_df, elevation_df, land_use_df, verbose=True)
    
    # Preprocess targets
    targets, blue_forces = fe.preprocess_target_data()
    
    # Create terrain cost map for vehicles
    terrain_cost = fe.create_terrain_cost_map('vehicle')
    
    # Create grid and add terrain
    grid_df, resolution = fe.create_grid_map()
    grid_with_terrain = fe.add_terrain_to_grid(grid_df, terrain_cost)
    
    # Create weighted cost surface
    weighted_cost = fe.create_weighted_cost_surface(grid_with_terrain, 'vehicle', blue_forces)
    
    # Create mobility network
    mobility_network = fe.create_mobility_network(weighted_cost, 'vehicle')
    
    # Create movement predictor
    predictor = MovementPredictor(
        targets, 
        blue_forces, 
        weighted_cost, 
        resolution,
        mobility_network,
        verbose=True
    )
    
    # Analyze target movement patterns
    target_id = targets['target_id'].iloc[0]
    target_analysis = predictor.historical_movement_analysis(target_id)
    
    # Generate predictions
    simple_pred = predictor.predict_next_position_simple(target_id, 30)
    terrain_pred = predictor.predict_with_terrain(target_id, 30)
    tactical_pred = predictor.predict_tactical_movement(target_id, 30)
    
    # Generate heatmap
    heatmap = predictor.generate_probability_heatmap(target_id, 30, 'integrated')
    
    print("Movement prediction complete.")