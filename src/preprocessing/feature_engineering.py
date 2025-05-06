"""
Feature Engineering Module

This module transforms raw data into features suitable for movement prediction models.
Key capabilities include:
- Preprocessing target observations
- Creating terrain cost surfaces
- Generating uniform spatial grids for predictions
- Extracting features for movement prediction models
- Military doctrine integration for tactical behavior modeling
"""

import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import griddata, RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, LineString, Polygon # type: ignore
from shapely.ops import nearest_points # type: ignore
import math
import datetime
import networkx as nx # type: ignore
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """
    Class for transforming raw data into features suitable for movement prediction.
    
    Attributes:
        targets_df (pd.DataFrame): DataFrame containing target observations
        elevation_df (pd.DataFrame): DataFrame containing elevation data
        land_use_df (pd.DataFrame): DataFrame containing land use data
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, targets_df, elevation_df, land_use_df, verbose=True):
        """
        Initialize the FeatureEngineering class.
        
        Args:
            targets_df (pd.DataFrame): DataFrame containing target observations
            elevation_df (pd.DataFrame): DataFrame containing elevation data
            land_use_df (pd.DataFrame): DataFrame containing land use data
            verbose (bool): Whether to print detailed information
        """
        self.targets_df = targets_df
        self.elevation_df = elevation_df
        self.land_use_df = land_use_df
        self.verbose = verbose
        
        # Mobility costs for different land use types by target class
        # Higher values indicate more difficult terrain to traverse
        self.mobility_costs = {
            'vehicle': {
                'urban': 2.0,     # Moderate difficulty due to obstacles, traffic
                'road': 1.0,      # Easiest for vehicles
                'forest': 3.5,    # Very difficult
                'open': 1.5,      # Fairly easy
                'water': 100.0,   # Impassable
                'wetland': 10.0,  # Very difficult
                'restricted': 5.0 # Difficult
            },
            'infantry': {
                'urban': 1.2,     # Good cover, but obstacles
                'road': 1.5,      # Easy but exposed
                'forest': 1.8,    # Slower but good cover
                'open': 1.3,      # Easy but exposed
                'water': 5.0,     # Difficult but possible for small water bodies
                'wetland': 2.5,   # Difficult
                'restricted': 3.0 # Difficult
            },
            'artillery': {
                'urban': 2.5,     # Difficult due to size
                'road': 1.2,      # Good for artillery transport
                'forest': 4.0,    # Very difficult
                'open': 1.8,      # Moderate
                'water': 100.0,   # Impassable
                'wetland': 8.0,   # Very difficult
                'restricted': 4.0 # Difficult
            },
            'command': {
                'urban': 1.8,     # Protection but limited mobility
                'road': 1.1,      # Good mobility
                'forest': 3.0,    # Limited mobility
                'open': 1.7,      # Exposed
                'water': 100.0,   # Impassable
                'wetland': 7.0,   # Very difficult
                'restricted': 3.5 # Difficult
            }
        }
        
        # Concealment values for different land use types (higher is better concealment)
        self.concealment = {
            'urban': 0.8,      # Good concealment in buildings
            'road': 0.1,       # Very exposed
            'forest': 0.9,     # Excellent concealment
            'open': 0.2,       # Poor concealment
            'water': 0.0,      # No concealment
            'wetland': 0.6,    # Good concealment
            'restricted': 0.7  # Variable concealment
        }
        
        # Military doctrine parameters
        self.doctrine_params = {
            # Default minimum spacing between units of same class (in km)
            'spacing': {
                'vehicle': 0.3,
                'infantry': 0.1,
                'artillery': 0.5,
                'command': 0.2
            },
            # Field of view angles (degrees from heading)
            'fov': {
                'vehicle': 120,
                'infantry': 180,
                'artillery': 90,
                'command': 360
            },
            # Blue force detection radius (km)
            'detection_radius': {
                'vehicle': 3.0,
                'infantry': 1.5,
                'artillery': 4.0,
                'command': 5.0
            },
            # Distance-to-blue-force factor (propensity to avoid)
            'blue_avoidance': {
                'vehicle': 0.8,
                'infantry': 0.7,
                'artillery': 0.9,
                'command': 0.8
            },
            # Concealment importance factor
            'concealment_factor': {
                'vehicle': 0.6,
                'infantry': 0.9,
                'artillery': 0.8,
                'command': 0.7
            },
            # Tactical movement patterns
            'movement_patterns': {
                'vehicle': 'canalize',     # Follows easiest path
                'infantry': 'disperse',    # Spreads out, uses cover
                'artillery': 'leapfrog',   # Move, set up, move
                'command': 'protected'     # Stays protected
            }
        }
        
        # Check if needed columns exist, otherwise add them
        if 'is_blue' not in self.targets_df.columns:
            self.targets_df['is_blue'] = 0
        
        if self.verbose:
            print("Feature Engineering initialized")
            
        # Preprocessing results cache
        self.terrain_cost_maps = {}
        self.grid_data = None
    
    def preprocess_target_data(self):
        """
        Preprocess target data and extract features.
        
        Calculates derived features like speed, heading, time differences, etc.
        
        Returns:
            tuple: (targets DataFrame, blue forces DataFrame)
        """
        df = self.targets_df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by target_id and timestamp
        df = df.sort_values(['target_id', 'timestamp'])
        
        # Calculate time differences between consecutive observations (in minutes)
        df['time_diff'] = df.groupby('target_id')['timestamp'].diff().dt.total_seconds() / 60
        
        # Calculate distance between consecutive observations
        df['lat_prev'] = df.groupby('target_id')['latitude'].shift(1)
        df['lon_prev'] = df.groupby('target_id')['longitude'].shift(1)
        
        # Haversine distance calculation (in kilometers)
        R = 6371  # Earth radius in kilometers
        df['lat_rad'] = np.radians(df['latitude'])
        df['lon_rad'] = np.radians(df['longitude'])
        df['lat_prev_rad'] = np.radians(df['lat_prev'])
        df['lon_prev_rad'] = np.radians(df['lon_prev'])
        
        df['dlon'] = df['lon_rad'] - df['lon_prev_rad']
        df['dlat'] = df['lat_rad'] - df['lat_prev_rad']
        
        a = np.sin(df['dlat']/2)**2 + np.cos(df['lat_prev_rad']) * np.cos(df['lat_rad']) * np.sin(df['dlon']/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        df['distance_km'] = R * c
        
        # Calculate speed if not already available (km/h)
        if 'speed' not in df.columns:
            df['speed'] = df['distance_km'] / (df['time_diff'] / 60)
            df['speed'] = df['speed'].fillna(0)
        
        # Calculate heading if not already available (degrees)
        if 'heading' not in df.columns:
            y = np.sin(df['dlon']) * np.cos(df['lat_rad'])
            x = np.cos(df['lat_prev_rad']) * np.sin(df['lat_rad']) - np.sin(df['lat_prev_rad']) * np.cos(df['lat_rad']) * np.cos(df['dlon'])
            df['heading'] = np.degrees((np.arctan2(y, x)) % (2 * np.pi))
            df['heading'] = df['heading'].fillna(0)
        
        # Handle NaN values for first observation of each target
        df['time_diff'] = df['time_diff'].fillna(0)
        df['distance_km'] = df['distance_km'].fillna(0)
        
        # Extract time-related features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Create day/night indicator (simplified, could be improved with actual sunrise/sunset data)
        df['is_night'] = ((df['hour'] >= 18) | (df['hour'] <= 6)).astype(int)
        
        # Calculate acceleration (change in speed between observations, km/h²)
        df['speed_prev'] = df.groupby('target_id')['speed'].shift(1).fillna(df['speed'])
        df['acceleration'] = (df['speed'] - df['speed_prev']) / (df['time_diff'] / 60).replace(0, 1)  # Per hour
        
        # Calculate rate of turn (degrees per minute)
        df['heading_prev'] = df.groupby('target_id')['heading'].shift(1).fillna(df['heading'])
        df['heading_diff'] = (df['heading'] - df['heading_prev'] + 180) % 360 - 180  # Handle circular values
        df['turn_rate'] = df['heading_diff'] / df['time_diff'].replace(0, 1)  # Degrees per minute
        
        # Calculate stop duration (how long a target has been stationary)
        df['is_moving'] = (df['speed'] > 0.5).astype(int)  # Threshold for "moving"
        df['stop_flag'] = (~df['is_moving'] & df.groupby('target_id')['is_moving'].shift(1).fillna(1).astype(bool)).astype(int)
        df['stop_id'] = df.groupby('target_id')['stop_flag'].cumsum()
        df['stop_duration'] = df.groupby(['target_id', 'stop_id'])['time_diff'].cumsum() * (~df['is_moving'])
        
        # Calculate terrain cost for each observation location (if we have terrain data)
        if not self.elevation_df.empty and not self.land_use_df.empty:
            # Normalize target coordinates to match elevation and land use coordinates
            lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
            lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
            
            df['x_norm'] = (df['longitude'] - lon_min) / (lon_max - lon_min)
            df['y_norm'] = (df['latitude'] - lat_min) / (lat_max - lat_min)
            
            # Get nearest elevation and land use points using a KD-tree
            terrain_df = pd.merge(self.elevation_df, self.land_use_df, on=['x', 'y'])
            terrain_points = terrain_df[['x', 'y']].values
            tree = cKDTree(terrain_points)
            
            # Query nearest points for each target observation
            target_points = df[['x_norm', 'y_norm']].values
            distances, indices = tree.query(target_points)
            
            # Add terrain information
            df['elevation'] = terrain_df.iloc[indices]['elevation'].values
            df['land_use'] = terrain_df.iloc[indices]['land_use'].values
            df['land_use_type'] = terrain_df.iloc[indices]['land_use_type'].values
            
            # Add terrain cost for each target based on its class
            df['terrain_cost'] = df.apply(
                lambda row: self.mobility_costs.get(
                    row['target_class'], self.mobility_costs['infantry']
                ).get(
                    row['land_use_type'], 1.0
                ),
                axis=1
            )
            
            # Add concealment value
            df['concealment'] = df['land_use_type'].map(self.concealment)
        
        # Extract proximity features (distance to key terrain features)
        if not self.land_use_df.empty:
            # Find roads, urban areas, and water bodies
            roads = self.land_use_df[self.land_use_df['land_use_type'] == 'road']
            urban = self.land_use_df[self.land_use_df['land_use_type'] == 'urban']
            water = self.land_use_df[self.land_use_df['land_use_type'] == 'water']
            forest = self.land_use_df[self.land_use_df['land_use_type'] == 'forest']
            
            # Function to calculate minimum distance to a feature
            def calc_min_distance(row, feature_df):
                if feature_df.empty:
                    return np.nan
                
                # Create KD-tree for quick nearest neighbor search
                feature_points = feature_df[['x', 'y']].values
                tree = cKDTree(feature_points)
                
                # Query nearest point
                distance, _ = tree.query([row['x_norm'], row['y_norm']])
                
                # Scale distance back to approximate km (very rough approximation)
                # Assumes x, y are normalized 0-1 and the area is approximately 10km x 10km
                lat_range = df['latitude'].max() - df['latitude'].min()
                lon_range = df['longitude'].max() - df['longitude'].min()
                
                # Approximate conversion using haversine formula
                R = 6371  # Earth radius in km
                dlat = lat_range * distance
                dlon = lon_range * distance
                a = np.sin(np.radians(dlat)/2)**2 + np.cos(np.radians(row['latitude'])) * np.cos(np.radians(row['latitude'] + dlat)) * np.sin(np.radians(dlon)/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return R * c
            
            # Calculate distances to key features
            if 'x_norm' in df.columns and 'y_norm' in df.columns:
                if not roads.empty:
                    df['dist_to_road'] = df.apply(lambda row: calc_min_distance(row, roads), axis=1)
                if not urban.empty:
                    df['dist_to_urban'] = df.apply(lambda row: calc_min_distance(row, urban), axis=1)
                if not water.empty:
                    df['dist_to_water'] = df.apply(lambda row: calc_min_distance(row, water), axis=1)
                if not forest.empty:
                    df['dist_to_forest'] = df.apply(lambda row: calc_min_distance(row, forest), axis=1)
        
        # Calculate blue force proximity and risk exposure
        if 'is_blue' in df.columns and 1 in df['is_blue'].values:
            blue_forces = df[df['is_blue'] == 1].copy()
            targets = df[df['is_blue'] == 0].copy()
            
            # For each target observation, calculate minimum distance to a blue force
            targets['blue_proximity'] = targets.apply(
                lambda row: self._calculate_blue_proximity(row, blue_forces), axis=1
            )
            
            # Calculate blue force field of view exposure
            targets['blue_exposure'] = targets.apply(
                lambda row: self._calculate_blue_exposure(row, blue_forces), axis=1
            )
            
            # Calculate concealment-adjusted exposure
            if 'concealment' in targets.columns:
                targets['effective_exposure'] = targets['blue_exposure'] * (1 - targets['concealment'])
            
            # Add blue force features back to the main dataframe
            df = pd.concat([
                targets,
                blue_forces
            ])
        
        # Extract formation features for groups of targets
        target_classes = df['target_class'].unique()
        for target_class in target_classes:
            # Skip blue forces
            if 'is_blue' in df.columns and df[df['target_class'] == target_class]['is_blue'].any():
                continue
                
            class_targets = df[df['target_class'] == target_class].copy()
            
            # Get unique timestamps
            timestamps = class_targets['timestamp'].unique()
            
            # For each timestamp, calculate formation features
            for ts in timestamps:
                targets_at_ts = class_targets[class_targets['timestamp'] == ts]
                
                if len(targets_at_ts) >= 3:  # Need at least 3 points for meaningful formation analysis
                    # Calculate centroid
                    centroid_lat = targets_at_ts['latitude'].mean()
                    centroid_lon = targets_at_ts['longitude'].mean()
                    
                    # Calculate distance from each target to centroid
                    dlat = np.radians(targets_at_ts['latitude'] - centroid_lat)
                    dlon = np.radians(targets_at_ts['longitude'] - centroid_lon)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(centroid_lat)) * np.cos(np.radians(targets_at_ts['latitude'])) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distances_to_centroid = 6371 * c  # km
                    
                    # Calculate formation dispersion (standard deviation of distances)
                    formation_dispersion = distances_to_centroid.std()
                    
                    # Calculate average nearest neighbor distance
                    points = targets_at_ts[['latitude', 'longitude']].values
                    if len(points) > 1:
                        tree = cKDTree(points)
                        distances, _ = tree.query(points, k=2)  # k=2 to get the nearest neighbor (excluding self)
                        avg_nn_distance = distances[:, 1].mean()  # Index 1 is the nearest neighbor (index 0 is self)
                    else:
                        avg_nn_distance = 0
                    
                    # Assign formation features to all targets at this timestamp
                    indices = class_targets[class_targets['timestamp'] == ts].index
                    df.loc[indices, 'formation_dispersion'] = formation_dispersion
                    df.loc[indices, 'avg_nn_distance'] = avg_nn_distance
        
        # Clean up intermediate columns
        cols_to_drop = ['lat_rad', 'lon_rad', 'lat_prev_rad', 'lon_prev_rad', 'dlon', 'dlat', 
                        'lat_prev', 'lon_prev', 'speed_prev', 'heading_prev', 'stop_flag']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Separate targets and blue forces
        blue_forces = df[df['is_blue'] == 1].copy() if 'is_blue' in df.columns else pd.DataFrame()
        targets = df[df['is_blue'] == 0].copy() if 'is_blue' in df.columns else df.copy()
        
        if self.verbose:
            print(f"Preprocessed {len(targets)} target observations")
            print(f"Extracted {len(blue_forces)} blue force observations")
            print(f"Generated {len(df.columns)} features")
        
        return targets, blue_forces
    
    def _calculate_blue_proximity(self, target_row, blue_forces_df):
        """
        Calculate the minimum distance to any blue force.
        
        Args:
            target_row (pd.Series): Row containing target data
            blue_forces_df (pd.DataFrame): DataFrame containing blue force data
            
        Returns:
            float: Minimum distance to any blue force (in km)
        """
        if blue_forces_df.empty:
            return np.inf
            
        # Filter blue forces to the same timestamp
        if 'timestamp' in target_row and 'timestamp' in blue_forces_df.columns:
            blue_at_time = blue_forces_df[blue_forces_df['timestamp'] <= target_row['timestamp']]
            
            # If no blue forces at or before this time, return infinity
            if blue_at_time.empty:
                return np.inf
            
            # Get the most recent position for each blue force
            blue_at_time = blue_at_time.sort_values('timestamp').groupby('blue_id').last().reset_index()
        else:
            blue_at_time = blue_forces_df
        
        # Calculate Haversine distance to each blue force
        R = 6371  # Earth radius in kilometers
        dlat = np.radians(blue_at_time['latitude'] - target_row['latitude'])
        dlon = np.radians(blue_at_time['longitude'] - target_row['longitude'])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(target_row['latitude'])) * np.cos(np.radians(blue_at_time['latitude'])) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = R * c
        
        # Return minimum distance
        return distances.min() if not distances.empty else np.inf
    
    def _calculate_blue_exposure(self, target_row, blue_forces_df):
        """
        Calculate exposure to blue forces based on their field of view.
        
        Args:
            target_row (pd.Series): Row containing target data
            blue_forces_df (pd.DataFrame): DataFrame containing blue force data
            
        Returns:
            float: Exposure score (0-1, higher means more exposed)
        """
        if blue_forces_df.empty:
            return 0.0
            
        # Filter blue forces to the same timestamp
        if 'timestamp' in target_row and 'timestamp' in blue_forces_df.columns:
            blue_at_time = blue_forces_df[blue_forces_df['timestamp'] <= target_row['timestamp']]
            
            # If no blue forces at or before this time, return 0
            if blue_at_time.empty:
                return 0.0
            
            # Get the most recent position for each blue force
            blue_at_time = blue_at_time.sort_values('timestamp').groupby('blue_id').last().reset_index()
        else:
            blue_at_time = blue_forces_df
        
        # Calculate distance and bearing to each blue force
        R = 6371  # Earth radius in kilometers
        exposure_scores = []
        
        for _, blue in blue_at_time.iterrows():
            # Calculate distance
            dlat = np.radians(blue['latitude'] - target_row['latitude'])
            dlon = np.radians(blue['longitude'] - target_row['longitude'])
            a = np.sin(dlat/2)**2 + np.cos(np.radians(target_row['latitude'])) * np.cos(np.radians(blue['latitude'])) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            # Calculate bearing from blue force to target
            y = np.sin(dlon) * np.cos(np.radians(target_row['latitude']))
            x = np.cos(np.radians(blue['latitude'])) * np.sin(np.radians(target_row['latitude'])) - \
                np.sin(np.radians(blue['latitude'])) * np.cos(np.radians(target_row['latitude'])) * np.cos(dlon)
            bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
            
            # Get blue force heading if available
            if 'heading' in blue:
                blue_heading = blue['heading']
                
                # Calculate difference between bearing and heading (target's position relative to blue's heading)
                angle_diff = abs((bearing - blue_heading + 180) % 360 - 180)
                
                # Calculate field of view factor (1.0 if directly in front, decreasing toward sides)
                # Assume a 120-degree effective field of view for blue forces
                fov = 120
                fov_factor = max(0, 1 - (angle_diff / (fov / 2)))
            else:
                # If no heading available, assume equal visibility in all directions
                fov_factor = 1.0
            
            # Determine detection range based on blue force class
            if 'blue_class' in blue:
                detection_range = self.doctrine_params['detection_radius'].get(blue['blue_class'], 2.0)
            else:
                detection_range = 2.0  # Default detection range in km
            
            # Calculate exposure based on distance (inverse square law) and field of view
            # Result is between 0 (not exposed) and 1 (fully exposed at zero distance)
            distance_factor = max(0, 1 - (distance / detection_range)**2) if distance < detection_range else 0
            exposure = distance_factor * fov_factor
            
            exposure_scores.append(exposure)
        
        # Return maximum exposure (most exposed to any single blue force)
        return max(exposure_scores) if exposure_scores else 0.0
    
    def calculate_terrain_gradient(self, elevation_grid):
        """
        Calculate the gradient (slope) of the terrain.
        
        Args:
            elevation_grid (numpy.ndarray): Grid of elevation values
            
        Returns:
            tuple: (gradient_y, gradient_x, gradient_magnitude)
        """
        # Calculate gradient using central differences
        gy, gx = np.gradient(elevation_grid)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        return gy, gx, gradient_magnitude
    
    def create_terrain_cost_map(self, target_class='infantry'):
        """
        Create a terrain cost map based on elevation and land use.
        
        Takes into account slope, land use, and target mobility characteristics.
        
        Args:
            target_class (str): Type of target ('vehicle', 'infantry', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with terrain cost information
        """
        # Check if already computed
        if target_class in self.terrain_cost_maps:
            return self.terrain_cost_maps[target_class]
            
        # Check if we have the required data
        if self.elevation_df.empty or self.land_use_df.empty:
            if self.verbose:
                print("Warning: Missing elevation or land use data for terrain cost map")
            return pd.DataFrame()
        
        # Combine elevation and land use data
        terrain_df = pd.merge(self.elevation_df, self.land_use_df, on=['x', 'y'])
        
        # Get dimensions and create grid
        x_values = sorted(terrain_df['x'].unique())
        y_values = sorted(terrain_df['y'].unique())
        
        # Check if we can create a proper grid
        if len(x_values) * len(y_values) != len(terrain_df):
            # Data is not on a regular grid, need to interpolate
            if self.verbose:
                print("Terrain data is not on a regular grid, interpolating...")
            
            # Create a regular grid
            x_grid = np.linspace(min(x_values), max(x_values), 100)
            y_grid = np.linspace(min(y_values), max(y_values), 100)
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Interpolate elevation
            points = terrain_df[['x', 'y']].values
            elevation_values = terrain_df['elevation'].values
            elevation_grid = griddata(points, elevation_values, (xx, yy), method='linear')
            
            # Interpolate land use (nearest neighbor to preserve categories)
            land_use_values = terrain_df['land_use'].values
            land_use_grid = griddata(points, land_use_values, (xx, yy), method='nearest')
            
            # Convert back to DataFrame
            terrain_df = pd.DataFrame({
                'x': xx.flatten(),
                'y': yy.flatten(),
                'elevation': elevation_grid.flatten(),
                'land_use': land_use_grid.flatten()
            })
            
            # Add land use type labels
            land_use_map = {i: t for i, t in zip(
                terrain_df['land_use'].unique(),
                terrain_df.merge(self.land_use_df[['land_use', 'land_use_type']], 
                                on='land_use', how='left')['land_use_type'].unique()
            )}
            terrain_df['land_use_type'] = terrain_df['land_use'].map(land_use_map)
        
        # Create elevation grid for gradient calculation
        elevation_pivot = terrain_df.pivot(index='y', columns='x', values='elevation')
        elevation_grid = elevation_pivot.values
        
        # Calculate slope (gradient of elevation)
        gy, gx, gradient_magnitude = self.calculate_terrain_gradient(elevation_grid)
        
        # Flatten back to DataFrame
        slope_flattened = gradient_magnitude.flatten()
        
        # Add slope to terrain DataFrame
        terrain_df['slope'] = slope_flattened
        
        # Scale slope to a reasonable range (0-1)
        max_slope = terrain_df['slope'].max()
        if max_slope > 0:
            terrain_df['slope_factor'] = terrain_df['slope'] / max_slope
        else:
            terrain_df['slope_factor'] = 0
            
        # Apply mobility costs based on land use
        terrain_df['mobility_cost'] = terrain_df['land_use_type'].map(
            lambda x: self.mobility_costs.get(target_class, self.mobility_costs['infantry']).get(x, 1.0)
        )
        
        # Apply slope factor to mobility cost
        # The effect of slope increases with the base mobility cost
        terrain_df['total_cost'] = terrain_df['mobility_cost'] * (1 + 2 * terrain_df['slope_factor'])
        
        # Add concealment values
        terrain_df['concealment'] = terrain_df['land_use_type'].map(self.concealment)
        
        # Calculate line-of-sight visibility index
        # This measures how visible a point is from surrounding terrain
        terrain_df['visibility_index'] = self._calculate_visibility_index(elevation_grid)
        
        # Calculate tactical value (combination of concealment and mobility)
        # For attackers: balance mobility and concealment
        terrain_df['tactical_value_attack'] = (
            (1 / terrain_df['total_cost']) * 0.7 + 
            terrain_df['concealment'] * 0.3
        )
        
        # For defenders: prioritize concealment and visibility
        terrain_df['tactical_value_defend'] = (
            terrain_df['concealment'] * 0.6 + 
            terrain_df['visibility_index'] * 0.4
        )
        
        # Cache the result
        self.terrain_cost_maps[target_class] = terrain_df
        
        if self.verbose:
            print(f"Created terrain cost map for {target_class}")
            print(f"Terrain cost range: {terrain_df['total_cost'].min():.2f} to {terrain_df['total_cost'].max():.2f}")
        
        return terrain_df
    
    def _calculate_visibility_index(self, elevation_grid):
        """
        Calculate a visibility index for each point in the terrain.
        
        Higher values indicate better visibility (good for observation).
        
        Args:
            elevation_grid (numpy.ndarray): Grid of elevation values
            
        Returns:
            numpy.ndarray: Visibility index values
        """
        # Get grid dimensions
        height, width = elevation_grid.shape
        
        # Calculate horizons in 8 directions
        horizons = np.zeros((height, width))
        
        # Define sampling directions (8 compass points)
        directions = [
            (0, 1),   # East
            (1, 1),   # Southeast
            (1, 0),   # South
            (1, -1),  # Southwest
            (0, -1),  # West
            (-1, -1), # Northwest
            (-1, 0),  # North
            (-1, 1)   # Northeast
        ]
        
        # Sample along each direction
        for di, dj in directions:
            # Create horizon map for this direction
            direction_horizon = np.zeros((height, width))
            
            # Maximum visible angle so far when traversing from edge
            for i in range(height):
                for j in range(width):
                    # Skip points outside grid
                    if not (0 <= i < height and 0 <= j < width):
                        continue
                        
                    # Check if this point is higher than previous maximum visible angle
                    max_angle = -np.inf
                    
                    # Trace from this point to edge in reverse direction
                    ni, nj = i, j
                    distance = 0
                    
                    while 0 <= ni < height and 0 <= nj < width:
                        # Calculate angle to this point
                        if distance > 0:
                            angle = (elevation_grid[ni, nj] - elevation_grid[i, j]) / distance
                            if angle > max_angle:
                                max_angle = angle
                                direction_horizon[i, j] += 1
                        
                        # Move to next point
                        ni -= di
                        nj -= dj
                        distance += 1
            
            # Normalize and add to total
            if direction_horizon.max() > 0:
                direction_horizon = direction_horizon / direction_horizon.max()
            horizons += direction_horizon
        
        # Normalize to 0-1 range
        if horizons.max() > 0:
            horizons = horizons / horizons.max()
        
        # Smooth the visibility index
        horizons = gaussian_filter(horizons, sigma=1.0)
        
        return horizons.flatten()
    
    def create_grid_map(self, resolution=100):
        """
        Create a uniform grid over the area of interest.
        
        The grid is used for generating predictions across the entire area.
        
        Args:
            resolution (int): Number of grid points in each dimension
            
        Returns:
            tuple: (grid DataFrame, resolution)
        """
        # Check if grid is already computed
        if self.grid_data is not None and self.grid_data[1] == resolution:
            return self.grid_data[0], resolution
            
        # Get bounds from target data
        lat_min, lat_max = self.targets_df['latitude'].min(), self.targets_df['latitude'].max()
        lon_min, lon_max = self.targets_df['longitude'].min(), self.targets_df['longitude'].max()
        
        # Add buffer
        lat_buffer = (lat_max - lat_min) * 0.1
        lon_buffer = (lon_max - lon_min) * 0.1
        lat_min -= lat_buffer
        lat_max += lat_buffer
        lon_min -= lon_buffer
        lon_max += lon_buffer
        
        # Create grid
        lat_grid = np.linspace(lat_min, lat_max, resolution)
        lon_grid = np.linspace(lon_min, lon_max, resolution)
        
        # Create all combinations
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Create dataframe
        grid_df = pd.DataFrame({
            'latitude': lat_mesh.flatten(),
            'longitude': lon_mesh.flatten(),
            'grid_id': range(resolution * resolution)
        })
        
        # Normalize to 0-1 range for matching with terrain data
        grid_df['x'] = (grid_df['longitude'] - lon_min) / (lon_max - lon_min)
        grid_df['y'] = (grid_df['latitude'] - lat_min) / (lat_max - lat_min)
        
        # Cache the result
        self.grid_data = (grid_df, resolution)
        
        if self.verbose:
            print(f"Created grid map with {len(grid_df)} points ({resolution}x{resolution})")
        
        return grid_df, resolution
    
    def add_terrain_to_grid(self, grid_df, terrain_df):
        """
        Add terrain information to the grid.
        
        Args:
            grid_df (pd.DataFrame): Grid DataFrame
            terrain_df (pd.DataFrame): Terrain DataFrame with cost information
            
        Returns:
            pd.DataFrame: Grid DataFrame with added terrain information
        """
        if terrain_df.empty:
            if self.verbose:
                print("Warning: No terrain data to add to grid")
            return grid_df
            
        # Create a KD-tree from the terrain points
        terrain_points = terrain_df[['x', 'y']].values
        tree = cKDTree(terrain_points)
        
        # Query nearest terrain points for each grid point
        grid_points = grid_df[['x', 'y']].values
        distances, indices = tree.query(grid_points)
        
        # Add terrain information to grid
        for col in terrain_df.columns:
            if col not in ['x', 'y']:
                grid_df[col] = terrain_df.iloc[indices][col].values
        
        if self.verbose:
            print(f"Added {len(terrain_df.columns) - 2} terrain attributes to grid")
        
        return grid_df
    
    def create_weighted_cost_surface(self, grid_df, target_class, blue_forces_df=None):
        """
        Create a weighted cost surface combining terrain, concealment, and blue force avoidance.
        
        Args:
            grid_df (pd.DataFrame): Grid DataFrame with terrain information
            target_class (str): Type of target ('vehicle', 'infantry', etc.)
            blue_forces_df (pd.DataFrame): DataFrame containing blue force positions
            
        Returns:
            pd.DataFrame: Grid DataFrame with weighted cost
        """
        result_df = grid_df.copy()
        
        # Ensure we have the required columns
        if 'total_cost' not in result_df.columns or 'concealment' not in result_df.columns:
            if self.verbose:
                print("Warning: Missing total_cost or concealment columns")
            return result_df
            
        # Get doctrine parameters for this target class
        concealment_factor = self.doctrine_params['concealment_factor'].get(target_class, 0.5)
        blue_avoidance = self.doctrine_params['blue_avoidance'].get(target_class, 0.7)
        
        # Add blue force influence if available
        if blue_forces_df is not None and not blue_forces_df.empty:
            # Get most recent position for each blue force
            latest_blue = blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            
            # Calculate blue force influence on each grid point
            result_df['blue_influence'] = result_df.apply(
                lambda row: self._calculate_blue_influence(row, latest_blue, target_class),
                axis=1
            )
        else:
            # No blue forces, zero influence
            result_df['blue_influence'] = 0.0
        
        # Calculate weighted cost
        # Lower is better (easier to traverse, better concealed, further from blue forces)
        result_df['weighted_cost'] = (
            result_df['total_cost'] * (1 - concealment_factor) +
            (1 - result_df['concealment']) * concealment_factor +
            result_df['blue_influence'] * blue_avoidance
        )
        
        # Add tactical priority (inverse of weighted cost)
        # Higher is better for movement
        result_df['tactical_priority'] = 1 / result_df['weighted_cost']
        
        # Normalize tactical priority to 0-1 range
        max_priority = result_df['tactical_priority'].max()
        if max_priority > 0:
            result_df['tactical_priority'] = result_df['tactical_priority'] / max_priority
        
        if self.verbose:
            print(f"Created weighted cost surface for {target_class}")
            print(f"Weighted cost range: {result_df['weighted_cost'].min():.2f} to {result_df['weighted_cost'].max():.2f}")
        
        return result_df
    
    def _calculate_blue_influence(self, grid_point, blue_forces_df, target_class):
        """
        Calculate the influence of blue forces on a grid point.
        
        Args:
            grid_point (pd.Series): Grid point data
            blue_forces_df (pd.DataFrame): DataFrame containing blue force positions
            target_class (str): Type of target ('vehicle', 'infantry', etc.)
            
        Returns:
            float: Blue force influence (0-1, higher means stronger influence)
        """
        if blue_forces_df.empty:
            return 0.0
            
        # Get detection radius for this target class
        detection_radius = self.doctrine_params['detection_radius'].get(target_class, 2.0)
        
        # Calculate distance to each blue force
        R = 6371  # Earth radius in kilometers
        influences = []
        
        for _, blue in blue_forces_df.iterrows():
            # Calculate Haversine distance
            dlat = np.radians(blue['latitude'] - grid_point['latitude'])
            dlon = np.radians(blue['longitude'] - grid_point['longitude'])
            a = np.sin(dlat/2)**2 + np.cos(np.radians(grid_point['latitude'])) * np.cos(np.radians(blue['latitude'])) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            # Calculate influence based on distance (inverse square law)
            # Influence decreases with square of distance, becomes 0 at detection radius
            if distance < detection_radius:
                influence = (1 - (distance / detection_radius)**2)
                
                # If we have concealment information, adjust influence
                if 'concealment' in grid_point:
                    influence *= (1 - grid_point['concealment'])
                
                influences.append(influence)
        
        # Return maximum influence from any blue force
        return max(influences) if influences else 0.0
    
    def create_mobility_network(self, grid_df, target_class, max_connections=8):
        """
        Create a graph network for pathfinding based on the weighted cost surface.
        
        Args:
            grid_df (pd.DataFrame): Grid DataFrame with weighted cost
            target_class (str): Type of target ('vehicle', 'infantry', etc.)
            max_connections (int): Maximum number of connections per node
            
        Returns:
            nx.Graph: NetworkX graph for pathfinding
        """
        # Ensure we have weighted cost
        if 'weighted_cost' not in grid_df.columns:
            if self.verbose:
                print("Warning: Missing weighted_cost column, creating default mobility network")
            return self._create_default_mobility_network(grid_df)
            
        # Create graph
        G = nx.Graph()
        
        # Create nodes
        for idx, row in grid_df.iterrows():
            G.add_node(
                row['grid_id'],
                pos=(row['longitude'], row['latitude']),
                cost=row['weighted_cost']
            )
        
        # Reshape points to grid
        grid_size = int(np.sqrt(len(grid_df)))
        grid_ids = grid_df['grid_id'].values.reshape(grid_size, grid_size)
        costs = grid_df['weighted_cost'].values.reshape(grid_size, grid_size)
        
        # Define neighbor offsets
        # 8-connectivity: horizontal, vertical, and diagonal neighbors
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Create edges between neighboring cells
        for i in range(grid_size):
            for j in range(grid_size):
                # Current node ID
                node_id = grid_ids[i, j]
                
                # Current cell cost
                current_cost = costs[i, j]
                
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    
                    # Check if neighbor is within bounds
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        # Neighbor node ID
                        neighbor_id = grid_ids[ni, nj]
                        
                        # Neighbor cost
                        neighbor_cost = costs[ni, nj]
                        
                        # Edge weight (average cost)
                        # Use higher cost for diagonal movement (√2 times higher)
                        if di != 0 and dj != 0:  # Diagonal
                            distance_factor = np.sqrt(2)
                        else:  # Horizontal or vertical
                            distance_factor = 1.0
                            
                        # Calculate edge weight (higher cost = higher weight)
                        edge_weight = ((current_cost + neighbor_cost) / 2) * distance_factor
                        
                        # Add edge
                        G.add_edge(node_id, neighbor_id, weight=edge_weight)
        
        # If we have fewer than max_connections per node, try to add more
        if max_connections > 8:
            self._add_long_range_connections(G, grid_df, target_class, max_connections)
        
        if self.verbose:
            print(f"Created mobility network with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        return G
    
    def _create_default_mobility_network(self, grid_df):
        """
        Create a simple grid network without cost information.
        
        Args:
            grid_df (pd.DataFrame): Grid DataFrame
            
        Returns:
            nx.Graph: NetworkX graph
        """
        # Create graph
        G = nx.Graph()
        
        # Create nodes
        for idx, row in grid_df.iterrows():
            G.add_node(
                row['grid_id'],
                pos=(row['longitude'], row['latitude']),
                cost=1.0  # Default cost
            )
        
        # Reshape points to grid
        grid_size = int(np.sqrt(len(grid_df)))
        grid_ids = grid_df['grid_id'].values.reshape(grid_size, grid_size)
        
        # Define neighbor offsets (8-connectivity)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Create edges
        for i in range(grid_size):
            for j in range(grid_size):
                # Current node ID
                node_id = grid_ids[i, j]
                
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    
                    # Check if neighbor is within bounds
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        # Neighbor node ID
                        neighbor_id = grid_ids[ni, nj]
                        
                        # Edge weight (distance)
                        if di != 0 and dj != 0:  # Diagonal
                            edge_weight = np.sqrt(2)
                        else:  # Horizontal or vertical
                            edge_weight = 1.0
                            
                        # Add edge
                        G.add_edge(node_id, neighbor_id, weight=edge_weight)
        
        return G
    
    def _add_long_range_connections(self, G, grid_df, target_class, max_connections):
        """
        Add longer-range connections to the graph based on terrain features.
        
        Args:
            G (nx.Graph): NetworkX graph
            grid_df (pd.DataFrame): Grid DataFrame
            target_class (str): Type of target
            max_connections (int): Maximum number of connections per node
            
        Returns:
            None (modifies G in place)
        """
        # Get movement pattern for this target class
        movement_pattern = self.doctrine_params['movement_patterns'].get(target_class, 'canalize')
        
        # Different connection strategies based on movement pattern
        if movement_pattern == 'canalize':
            # For vehicles: add connections along roads and open terrain
            if 'land_use_type' in grid_df.columns:
                road_nodes = [
                    row['grid_id'] for _, row in grid_df.iterrows()
                    if row['land_use_type'] in ['road', 'open']
                ]
                
                for node_id in road_nodes:
                    # Find nearby road nodes
                    node_pos = G.nodes[node_id]['pos']
                    
                    # Find other road nodes within a certain distance
                    nearby_roads = [
                        other_id for other_id in road_nodes
                        if other_id != node_id and
                        self._calculate_distance(node_pos, G.nodes[other_id]['pos']) < 0.05  # ~5km approx
                    ]
                    
                    # Add connections
                    for other_id in nearby_roads[:3]:  # Limit to 3 additional connections
                        if not G.has_edge(node_id, other_id):
                            # Calculate edge weight
                            distance = self._calculate_distance(node_pos, G.nodes[other_id]['pos'])
                            edge_weight = distance * (G.nodes[node_id]['cost'] + G.nodes[other_id]['cost']) / 2
                            
                            G.add_edge(node_id, other_id, weight=edge_weight)
                            
        elif movement_pattern == 'disperse':
            # For infantry: add connections through forest and urban areas
            if 'land_use_type' in grid_df.columns:
                cover_nodes = [
                    row['grid_id'] for _, row in grid_df.iterrows()
                    if row['land_use_type'] in ['forest', 'urban']
                ]
                
                for node_id in cover_nodes:
                    # Find nearby cover nodes
                    node_pos = G.nodes[node_id]['pos']
                    
                    # Find other cover nodes within a certain distance
                    nearby_cover = [
                        other_id for other_id in cover_nodes
                        if other_id != node_id and
                        self._calculate_distance(node_pos, G.nodes[other_id]['pos']) < 0.02  # ~2km approx
                    ]
                    
                    # Add connections
                    for other_id in nearby_cover[:5]:  # Limit to 5 additional connections
                        if not G.has_edge(node_id, other_id):
                            # Calculate edge weight
                            distance = self._calculate_distance(node_pos, G.nodes[other_id]['pos'])
                            edge_weight = distance * (G.nodes[node_id]['cost'] + G.nodes[other_id]['cost']) / 2
                            
                            G.add_edge(node_id, other_id, weight=edge_weight)
                            
        elif movement_pattern in ['leapfrog', 'protected']:
            # For artillery/command: add connections to good defensive positions
            if 'tactical_value_defend' in grid_df.columns:
                # Find nodes with good defensive value
                defend_nodes = grid_df.nlargest(100, 'tactical_value_defend')['grid_id'].tolist()
                
                for node_id in defend_nodes:
                    # Find other defensive nodes within a certain distance
                    node_pos = G.nodes[node_id]['pos']
                    
                    # Find other defensive nodes within a certain distance
                    nearby_defend = [
                        other_id for other_id in defend_nodes
                        if other_id != node_id and
                        self._calculate_distance(node_pos, G.nodes[other_id]['pos']) < 0.03  # ~3km approx
                    ]
                    
                    # Add connections
                    for other_id in nearby_defend[:4]:  # Limit to 4 additional connections
                        if not G.has_edge(node_id, other_id):
                            # Calculate edge weight
                            distance = self._calculate_distance(node_pos, G.nodes[other_id]['pos'])
                            edge_weight = distance * (G.nodes[node_id]['cost'] + G.nodes[other_id]['cost']) / 2
                            
                            G.add_edge(node_id, other_id, weight=edge_weight)
    
    def _calculate_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1 (tuple): First position (longitude, latitude)
            pos2 (tuple): Second position (longitude, latitude)
            
        Returns:
            float: Distance
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_key_terrain_features(self, grid_df):
        """
        Identify key terrain features for tactical analysis.
        
        Args:
            grid_df (pd.DataFrame): Grid DataFrame with terrain information
            
        Returns:
            dict: Dictionary of key terrain features
        """
        # Check if we have the required columns
        if not all(col in grid_df.columns for col in ['elevation', 'land_use_type', 'visibility_index']):
            if self.verbose:
                print("Warning: Missing columns for key terrain analysis")
            return {}
            
        # Create key terrain dictionary
        key_terrain = {}
        
        # Find high ground (highest 5% of elevation)
        high_ground = grid_df.nlargest(int(len(grid_df) * 0.05), 'elevation')
        key_terrain['high_ground'] = high_ground
        
        # Find good observation points (highest 5% of visibility index)
        observation_points = grid_df.nlargest(int(len(grid_df) * 0.05), 'visibility_index')
        key_terrain['observation_points'] = observation_points
        
        # Find chokepoints (narrow passages based on terrain costs)
        # For simplicity, find points where adjacent cells have high cost differences
        if 'total_cost' in grid_df.columns:
            # Reshape to grid for neighbor analysis
            grid_size = int(np.sqrt(len(grid_df)))
            cost_grid = grid_df['total_cost'].values.reshape(grid_size, grid_size)
            
            # Calculate local cost gradient
            gy, gx = np.gradient(cost_grid)
            cost_gradient = np.sqrt(gx**2 + gy**2)
            
            # Find high gradient points (potential chokepoints)
            threshold = np.percentile(cost_gradient, 95)
            chokepoint_mask = cost_gradient > threshold
            
            # Get grid IDs of chokepoints
            chokepoint_ids = grid_df['grid_id'].values.reshape(grid_size, grid_size)[chokepoint_mask]
            
            # Create DataFrame of chokepoints
            chokepoints = grid_df[grid_df['grid_id'].isin(chokepoint_ids)]
            key_terrain['chokepoints'] = chokepoints
        
        # Find concealed routes
        # Find paths with high concealment between major terrain features
        if 'concealment' in grid_df.columns:
            # Find areas with good concealment
            concealed_areas = grid_df[grid_df['concealment'] > 0.7]
            
            # Identify potential concealed routes
            # For simplicity, find continuous areas of high concealment
            concealed_routes = grid_df[grid_df['concealment'] > 0.5]
            key_terrain['concealed_routes'] = concealed_routes
        
        # Find avenue of approach (easy movement corridors)
        if 'weighted_cost' in grid_df.columns:
            # Find areas with low weighted cost
            avenues = grid_df[grid_df['weighted_cost'] < grid_df['weighted_cost'].quantile(0.2)]
            key_terrain['avenues_of_approach'] = avenues
        
        if self.verbose:
            print("Identified key terrain features:")
            for feature, data in key_terrain.items():
                print(f"  {feature}: {len(data)} points")
        
        return key_terrain
    
    def export_features(self, output_folder="./features"):
        """
        Export processed features to files for later use.
        
        Args:
            output_folder (str): Path to output folder
            
        Returns:
            bool: Success indicator
        """
        try:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Export processed targets
            targets, blue_forces = self.preprocess_target_data()
            targets.to_csv(os.path.join(output_folder, "processed_targets.csv"), index=False)
            if not blue_forces.empty:
                blue_forces.to_csv(os.path.join(output_folder, "processed_blue_forces.csv"), index=False)
            
            # Export terrain cost maps for each target class
            for target_class in self.mobility_costs.keys():
                terrain_cost = self.create_terrain_cost_map(target_class)
                terrain_cost.to_csv(os.path.join(output_folder, f"terrain_cost_{target_class}.csv"), index=False)
            
            # Export grid map
            grid_df, resolution = self.create_grid_map()
            grid_df.to_csv(os.path.join(output_folder, "grid_map.csv"), index=False)
            
            # Export weighted cost surface for each target class
            for target_class in self.mobility_costs.keys():
                terrain_cost = self.create_terrain_cost_map(target_class)
                grid_with_terrain = self.add_terrain_to_grid(grid_df, terrain_cost)
                weighted_cost = self.create_weighted_cost_surface(grid_with_terrain, target_class, blue_forces)
                weighted_cost.to_csv(os.path.join(output_folder, f"weighted_cost_{target_class}.csv"), index=False)
            
            # Export key terrain features
            terrain_cost = self.create_terrain_cost_map('infantry')  # Use infantry as default
            grid_with_terrain = self.add_terrain_to_grid(grid_df, terrain_cost)
            key_terrain = self.calculate_key_terrain_features(grid_with_terrain)
            
            for feature, data in key_terrain.items():
                data.to_csv(os.path.join(output_folder, f"key_terrain_{feature}.csv"), index=False)
            
            # Export metadata
            with open(os.path.join(output_folder, "metadata.txt"), 'w') as f:
                f.write(f"Features generated on: {datetime.datetime.now()}\n")
                f.write(f"Target observations: {len(targets)}\n")
                f.write(f"Blue force observations: {len(blue_forces)}\n")
                f.write(f"Grid resolution: {resolution}x{resolution}\n")
                f.write(f"Target classes: {targets['target_class'].unique()}\n")
                f.write(f"Time range: {targets['timestamp'].min()} to {targets['timestamp'].max()}\n")
            
            if self.verbose:
                print(f"Exported features to {output_folder}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error exporting features: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
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
    G = fe.create_mobility_network(weighted_cost, 'vehicle')
    
    # Calculate key terrain features
    key_terrain = fe.calculate_key_terrain_features(grid_with_terrain)
    
    # Export features
    fe.export_features()
    
    print("Feature engineering complete.")