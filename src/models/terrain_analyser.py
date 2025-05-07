"""
Terrain Analyser Module

This module provides analytical capabilities for terrain assessment and its impact
on target movement prediction. It integrates terrain data with tactical concepts
to evaluate mobility, concealment, fields of fire, and other terrain attributes.

Key capabilities:
- Terrain classification and feature extraction
- Cost surface generation for different target types
- Line-of-sight and visibility analysis
- Key terrain identification
- Mobility corridor analysis
- Concealment and cover assessment
- Terrain impact on tactical decision points
"""

import numpy as np
import pandas as pd # type: ignore
from scipy.ndimage import gaussian_filter, sobel, label
from scipy.spatial import cKDTree, Voronoi
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx # type: ignore
import math
import os
import pickle
from collections import defaultdict
from tqdm import tqdm # type: ignore
import warnings
warnings.filterwarnings('ignore')


class TerrainAnalyser:
    """
    Main class for terrain analysis, feature extraction, and terrain-based predictions.
    
    Attributes:
        elevation_grid (pd.DataFrame): DataFrame containing elevation data
        land_use_grid (pd.DataFrame): DataFrame containing land use data
        terrain_grid (pd.DataFrame): Combined terrain data with features
        grid_resolution (int): Resolution of the terrain grid
        terrain_features (dict): Dictionary of extracted terrain features
        config (dict): Configuration parameters
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, elevation_grid=None, land_use_grid=None, terrain_grid=None, 
                 grid_resolution=100, config=None, verbose=True):
        """
        Initialize the TerrainAnalyser.
        
        Args:
            elevation_grid (pd.DataFrame): DataFrame containing elevation data
            land_use_grid (pd.DataFrame): DataFrame containing land use data
            terrain_grid (pd.DataFrame): Combined terrain data (optional)
            grid_resolution (int): Resolution of the terrain grid
            config (dict): Configuration parameters (optional)
            verbose (bool): Whether to print detailed information
        """
        self.elevation_grid = elevation_grid
        self.land_use_grid = land_use_grid
        self.terrain_grid = terrain_grid
        self.grid_resolution = grid_resolution
        self.verbose = verbose
        
        # Default configuration
        self.default_config = {
            'slope_threshold': 30,  # Maximum traversable slope (degrees)
            'steep_slope': 20,      # Threshold for steep slope (degrees)
            'water_depth_threshold': 1.0,  # Maximum fordable water depth (meters)
            'concealment_distance': 300,  # Distance at which concealment is effective (meters)
            'observation_range': 3000,  # Maximum observation range (meters)
            'mobility_classifications': {
                'vehicle': {
                    'road': 0.1,       # Best mobility
                    'urban': 0.5,
                    'open_ground': 0.2,
                    'light_vegetation': 0.3,
                    'forest': 0.7,
                    'water': 0.9,      # Almost impassible
                    'steep_slope': 0.8  # Very difficult
                },
                'infantry': {
                    'road': 0.2,
                    'urban': 0.3,
                    'open_ground': 0.3,
                    'light_vegetation': 0.4,
                    'forest': 0.5,
                    'water': 0.9,
                    'steep_slope': 0.6
                },
                'artillery': {
                    'road': 0.2,
                    'urban': 0.6,
                    'open_ground': 0.4,
                    'light_vegetation': 0.5,
                    'forest': 0.8,
                    'water': 0.95,
                    'steep_slope': 0.9
                },
                'command': {
                    'road': 0.1,
                    'urban': 0.4,
                    'open_ground': 0.3,
                    'light_vegetation': 0.4,
                    'forest': 0.7,
                    'water': 0.9,
                    'steep_slope': 0.8
                }
            },
            'concealment_values': {
                'road': 0.1,
                'urban': 0.8,
                'open_ground': 0.1,
                'light_vegetation': 0.5,
                'forest': 0.9,
                'water': 0.2,
                'steep_slope': 0.4
            },
            'cover_values': {
                'road': 0.0,
                'urban': 0.7,
                'open_ground': 0.0,
                'light_vegetation': 0.2,
                'forest': 0.6,
                'water': 0.1,
                'steep_slope': 0.5
            },
            'key_terrain_factors': {
                'elevation_importance': 0.4,
                'observation_importance': 0.3,
                'mobility_importance': 0.2,
                'concealment_importance': 0.1
            },
            'cache_dir': './cache',  # Directory for caching analysis results
            'cache_terrain': True    # Whether to cache terrain analysis results
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Create cache directory if caching is enabled
        if self.config['cache_terrain']:
            os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Initialize terrain features
        self.terrain_features = {}
        
        # Initialize spatial index
        self.spatial_index = None
        
        # Coordinate bounds
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        
        # Check if terrain grid is provided
        if self.terrain_grid is not None:
            # Extract bounds
            self._extract_coordinate_bounds()
            
            # Create spatial index
            self._create_spatial_index()
            
            if self.verbose:
                print("Terrain Analyser initialized with terrain grid")
                print(f"Grid resolution: {self.grid_resolution}")
                print(f"Coordinate bounds: ({self.lat_min}, {self.lon_min}) to ({self.lat_max}, {self.lon_max})")
        elif self.elevation_grid is not None and self.land_use_grid is not None:
            # Create integrated terrain grid
            self.create_terrain_grid()
            
            if self.verbose:
                print("Terrain Analyser initialized and terrain grid created")
                print(f"Grid resolution: {self.grid_resolution}")
                print(f"Coordinate bounds: ({self.lat_min}, {self.lon_min}) to ({self.lat_max}, {self.lon_max})")
        else:
            if self.verbose:
                print("Terrain Analyser initialized without terrain data")
                print("Use 'create_terrain_grid()' to create terrain grid once data is loaded")
    
    def _extract_coordinate_bounds(self):
        """
        Extract coordinate bounds from terrain grid.
        """
        if self.terrain_grid is None or self.terrain_grid.empty:
            return
        
        # Extract bounds
        self.lat_min = self.terrain_grid['latitude'].min()
        self.lat_max = self.terrain_grid['latitude'].max()
        self.lon_min = self.terrain_grid['longitude'].min()
        self.lon_max = self.terrain_grid['longitude'].max()
    
    def _create_spatial_index(self):
        """
        Create spatial index for efficient spatial queries.
        """
        if self.terrain_grid is None or self.terrain_grid.empty:
            return
        
        # Create KD-tree for nearest neighbor searches
        self.spatial_index = cKDTree(self.terrain_grid[['latitude', 'longitude']].values)
        
        if self.verbose:
            print("Created spatial index for terrain grid")
    
    def create_terrain_grid(self):
        """
        Create integrated terrain grid from elevation and land use data.
        
        Returns:
            pd.DataFrame: Integrated terrain grid
        """
        # Check if we have the necessary data
        if self.elevation_grid is None:
            if self.verbose:
                print("Error: No elevation data provided")
            return None
        
        if self.land_use_grid is None:
            if self.verbose:
                print("Warning: No land use data provided, using elevation only")
        
        # Check for cached terrain grid
        cache_file = os.path.join(self.config['cache_dir'], 'terrain_grid.pkl')
        
        if self.config['cache_terrain'] and os.path.exists(cache_file):
            try:
                if self.verbose:
                    print(f"Loading cached terrain grid from {cache_file}")
                
                self.terrain_grid = pd.read_pickle(cache_file)
                
                # Extract bounds
                self._extract_coordinate_bounds()
                
                # Create spatial index
                self._create_spatial_index()
                
                return self.terrain_grid
            except:
                if self.verbose:
                    print("Failed to load cached terrain grid, recreating")
        
        if self.verbose:
            print("Creating integrated terrain grid...")
        
        # Start with the elevation grid
        terrain_grid = self.elevation_grid.copy()
        
        # Extract bounds
        self.lat_min = terrain_grid['latitude'].min()
        self.lat_max = terrain_grid['latitude'].max()
        self.lon_min = terrain_grid['longitude'].min()
        self.lon_max = terrain_grid['longitude'].max()
        
        # Generate a regular grid if not already
        if len(terrain_grid) != self.grid_resolution * self.grid_resolution:
            if self.verbose:
                print("Regridding to consistent resolution")
            
            # Create regular grid
            lat_vals = np.linspace(self.lat_min, self.lat_max, self.grid_resolution)
            lon_vals = np.linspace(self.lon_min, self.lon_max, self.grid_resolution)
            
            grid_points = []
            for lat in lat_vals:
                for lon in lon_vals:
                    grid_points.append({
                        'latitude': lat,
                        'longitude': lon,
                        'grid_id': len(grid_points)
                    })
            
            # Create new grid dataframe
            new_grid = pd.DataFrame(grid_points)
            
            # Interpolate elevation to new grid
            # Creating a function for elevation interpolation
            from scipy.interpolate import LinearNDInterpolator
            
            points = self.elevation_grid[['latitude', 'longitude']].values
            values = self.elevation_grid['elevation'].values
            
            # Create interpolator
            interp = LinearNDInterpolator(points, values)
            
            # Apply interpolation
            new_points = new_grid[['latitude', 'longitude']].values
            new_grid['elevation'] = interp(new_points)
            
            # Fill any NaN values with nearest neighbor
            if new_grid['elevation'].isna().any():
                from scipy.interpolate import NearestNDInterpolator
                
                # Create nearest interpolator
                nearest_interp = NearestNDInterpolator(points, values)
                
                # Get indices of NaN values
                na_indices = new_grid[new_grid['elevation'].isna()].index
                
                # Fill NaN values
                new_grid.loc[na_indices, 'elevation'] = nearest_interp(
                    new_grid.loc[na_indices, ['latitude', 'longitude']].values)
            
            # Replace terrain grid
            terrain_grid = new_grid
        
        # Add slope information
        # Reshape to 2D grid for gradient calculation
        elev_reshape = terrain_grid['elevation'].values.reshape(self.grid_resolution, self.grid_resolution)
        
        # Calculate gradient in x and y directions
        from scipy.ndimage import sobel
        
        # Estimate grid cell size in meters
        # Convert degrees to meters (approximately)
        lat_cell_size = (self.lat_max - self.lat_min) / (self.grid_resolution - 1) * 111320  # m/deg at equator
        lon_cell_size = (self.lon_max - self.lon_min) / (self.grid_resolution - 1) * 111320 * np.cos(np.radians((self.lat_max + self.lat_min) / 2))
        
        # Calculate gradient (meters/cell)
        grad_x = sobel(elev_reshape, axis=1) / (2 * lon_cell_size)
        grad_y = sobel(elev_reshape, axis=0) / (2 * lat_cell_size)
        
        # Calculate slope (in degrees)
        slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
        
        # Calculate aspect (direction of steepest descent)
        aspect = np.degrees(np.arctan2(-grad_y, -grad_x)) % 360
        
        # Reshape back to 1D and add to terrain grid
        terrain_grid['slope'] = slope.flatten()
        terrain_grid['aspect'] = aspect.flatten()
        
        # Identify steep slopes
        terrain_grid['is_steep_slope'] = terrain_grid['slope'] > self.config['steep_slope']
        
        # Add land use information if available
        if self.land_use_grid is not None:
            # Create a KD-tree for efficient nearest neighbor search
            land_use_tree = cKDTree(self.land_use_grid[['latitude', 'longitude']].values)
            
            # Find nearest land use point for each terrain grid point
            distances, indices = land_use_tree.query(terrain_grid[['latitude', 'longitude']].values, k=1)
            
            # Add land use information
            terrain_grid['land_use_type'] = self.land_use_grid.iloc[indices]['land_use_type'].values
            
            # Add any other land use attributes
            for col in self.land_use_grid.columns:
                if col not in ['latitude', 'longitude', 'grid_id']:
                    try:
                        terrain_grid[col] = self.land_use_grid.iloc[indices][col].values
                    except:
                        pass
        
        # Calculate terrain costs for different target types
        for target_type in self.config['mobility_classifications'].keys():
            cost_col = f'{target_type}_cost'
            terrain_grid[cost_col] = self.calculate_terrain_cost(terrain_grid, target_type)
        
        # Calculate concealment values
        terrain_grid['concealment'] = self.calculate_concealment(terrain_grid)
        
        # Calculate cover values
        terrain_grid['cover'] = self.calculate_cover(terrain_grid)
        
        # Add distance to roads if road information is available
        if 'land_use_type' in terrain_grid.columns:
            terrain_grid['dist_to_road'] = self.calculate_distance_to_feature(terrain_grid, 'road')
        
        # Save terrain grid
        self.terrain_grid = terrain_grid
        
        # Create spatial index
        self._create_spatial_index()
        
        # Cache the terrain grid if caching is enabled
        if self.config['cache_terrain']:
            try:
                self.terrain_grid.to_pickle(cache_file)
                if self.verbose:
                    print(f"Cached terrain grid to {cache_file}")
            except:
                if self.verbose:
                    print(f"Failed to cache terrain grid")
        
        if self.verbose:
            print(f"Created terrain grid with {len(terrain_grid)} points")
            print(f"Grid includes: elevation, slope, aspect, land use, concealment, cover, and terrain costs")
        
        return self.terrain_grid
    
    def calculate_terrain_cost(self, grid_df, target_type):
        """
        Calculate terrain cost for a specific target type.
        
        Args:
            grid_df (pd.DataFrame): Terrain grid dataframe
            target_type (str): Target type (vehicle, infantry, artillery, command)
            
        Returns:
            numpy.ndarray: Array of terrain costs
        """
        # Get mobility classifications for this target type
        mobility_class = self.config['mobility_classifications'].get(target_type, 
                                                                    self.config['mobility_classifications']['infantry'])
        
        # Initialize cost array (default to high cost)
        costs = np.ones(len(grid_df)) * 0.5
        
        # Apply land use based costs if available
        if 'land_use_type' in grid_df.columns:
            for land_type, cost_value in mobility_class.items():
                if land_type == 'steep_slope':
                    continue  # Handle slope separately
                
                # Find cells with this land type
                mask = grid_df['land_use_type'] == land_type
                costs[mask] = cost_value
        
        # Apply slope-based costs
        if 'slope' in grid_df.columns:
            # Find steep slope areas
            steep_mask = grid_df['slope'] > self.config['steep_slope']
            
            # Apply steep slope cost
            costs[steep_mask] = mobility_class.get('steep_slope', 0.8)
            
            # Find impassable slope areas
            impassable_mask = grid_df['slope'] > self.config['slope_threshold']
            
            # Set impassable areas to maximum cost
            costs[impassable_mask] = 0.95
        
        # Apply water-based costs if available
        if 'water_depth' in grid_df.columns:
            # Find deep water areas
            deep_water_mask = grid_df['water_depth'] > self.config['water_depth_threshold']
            
            # Set deep water to maximum cost
            costs[deep_water_mask] = 0.95
        
        return costs
    
    def calculate_concealment(self, grid_df):
        """
        Calculate concealment values for the terrain grid.
        
        Args:
            grid_df (pd.DataFrame): Terrain grid dataframe
            
        Returns:
            numpy.ndarray: Array of concealment values (0-1)
        """
        # Initialize concealment array (default to low concealment)
        concealment = np.ones(len(grid_df)) * 0.1
        
        # Apply land use based concealment if available
        if 'land_use_type' in grid_df.columns:
            for land_type, conceal_value in self.config['concealment_values'].items():
                # Find cells with this land type
                mask = grid_df['land_use_type'] == land_type
                concealment[mask] = conceal_value
        
        # Enhance concealment in areas with steep slopes
        if 'slope' in grid_df.columns:
            # Find steep areas that provide additional concealment
            slope_enhance_mask = (grid_df['slope'] > self.config['steep_slope'] / 2) & (grid_df['slope'] <= self.config['steep_slope'])
            
            # Increase concealment in these areas
            concealment[slope_enhance_mask] = np.maximum(concealment[slope_enhance_mask] + 0.2, 0.9)
        
        # Apply elevation-based concealment enhancement
        # Areas in depressions or behind hills get better concealment
        if 'elevation' in grid_df.columns:
            try:
                # Reshape to 2D grid
                elev_reshape = grid_df['elevation'].values.reshape(self.grid_resolution, self.grid_resolution)
                
                # Calculate local elevation variance
                from scipy.ndimage import generic_filter
                
                # Function to calculate elevation range in a window
                def elev_range(x):
                    return np.max(x) - np.min(x)
                
                # Calculate elevation range in 3x3 windows
                elev_var = generic_filter(elev_reshape, elev_range, size=3)
                
                # Areas with high local elevation variance get better concealment
                elev_var_flat = elev_var.flatten()
                high_var_mask = elev_var_flat > np.percentile(elev_var_flat, 70)
                
                # Increase concealment in high variance areas
                concealment[high_var_mask] = np.minimum(concealment[high_var_mask] + 0.15, 1.0)
            except:
                # Skip this enhancement if reshaping fails
                pass
        
        return concealment
    
    def calculate_cover(self, grid_df):
        """
        Calculate cover values for the terrain grid.
        
        Args:
            grid_df (pd.DataFrame): Terrain grid dataframe
            
        Returns:
            numpy.ndarray: Array of cover values (0-1)
        """
        # Initialize cover array (default to no cover)
        cover = np.zeros(len(grid_df))
        
        # Apply land use based cover if available
        if 'land_use_type' in grid_df.columns:
            for land_type, cover_value in self.config['cover_values'].items():
                # Find cells with this land type
                mask = grid_df['land_use_type'] == land_type
                cover[mask] = cover_value
        
        # Enhance cover in areas with steep slopes
        if 'slope' in grid_df.columns:
            # Find steep areas
            steep_mask = grid_df['slope'] > self.config['steep_slope']
            
            # Increase cover in steep areas
            cover[steep_mask] = np.maximum(cover[steep_mask], self.config['cover_values'].get('steep_slope', 0.5))
        
        # Enhance cover on reverse slopes (away from reference direction)
        # This simulates cover from enemy observation
        if 'aspect' in grid_df.columns:
            # Assume threat direction is from the north (0 degrees)
            # Areas facing away from threat (135-225 degrees) get better cover
            south_facing_mask = (grid_df['aspect'] > 135) & (grid_df['aspect'] < 225)
            
            # Increase cover for south-facing slopes
            cover[south_facing_mask] = np.minimum(cover[south_facing_mask] + 0.2, 1.0)
        
        return cover
    
    def calculate_distance_to_feature(self, grid_df, feature_type):
        """
        Calculate distance to nearest feature of specified type.
        
        Args:
            grid_df (pd.DataFrame): Terrain grid dataframe
            feature_type (str): Feature type to find (e.g., 'road', 'urban')
            
        Returns:
            numpy.ndarray: Array of distances (in km)
        """
        # Initialize distances to infinity
        distances = np.ones(len(grid_df)) * np.inf
        
        # Check if land use information is available
        if 'land_use_type' not in grid_df.columns:
            return distances
        
        # Find points with the feature type
        feature_points = grid_df[grid_df['land_use_type'] == feature_type]
        
        if feature_points.empty:
            return distances
        
        # Create a KD-tree for efficient nearest neighbor search
        feature_tree = cKDTree(feature_points[['latitude', 'longitude']].values)
        
        # Find nearest feature point for each grid point
        dist, _ = feature_tree.query(grid_df[['latitude', 'longitude']].values, k=1)
        
        # Convert degrees to kilometers (approximately)
        center_lat = (grid_df['latitude'].min() + grid_df['latitude'].max()) / 2
        km_per_degree = 111.32  # at equator
        km_per_degree_lon = km_per_degree * np.cos(np.radians(center_lat))
        
        # Calculate average km per degree for this region
        km_per_degree_avg = (km_per_degree + km_per_degree_lon) / 2
        
        # Convert to kilometers
        distances = dist * km_per_degree_avg
        
        return distances
    
    def create_cost_surface(self, target_type, include_tactical=True):
        """
        Create a total cost surface for the specified target type.
        
        Args:
            target_type (str): Target type (vehicle, infantry, artillery, command)
            include_tactical (bool): Include tactical factors like concealment
            
        Returns:
            pd.DataFrame: Terrain grid with total cost surface
        """
        if self.terrain_grid is None:
            if self.verbose:
                print("Error: Terrain grid not created yet")
                print("Call create_terrain_grid() first")
            return None
        
        # Check for cached cost surface
        cache_file = os.path.join(self.config['cache_dir'], f'cost_surface_{target_type}.pkl')
        
        if self.config['cache_terrain'] and os.path.exists(cache_file):
            try:
                if self.verbose:
                    print(f"Loading cached cost surface for {target_type} from {cache_file}")
                
                return pd.read_pickle(cache_file)
            except:
                if self.verbose:
                    print("Failed to load cached cost surface, recreating")
        
        if self.verbose:
            print(f"Creating cost surface for {target_type}")
        
        # Create a copy of the terrain grid
        cost_surface = self.terrain_grid.copy()
        
        # Get terrain cost column
        cost_col = f'{target_type}_cost'
        
        if cost_col not in cost_surface.columns:
            # Calculate terrain cost if not already present
            cost_surface[cost_col] = self.calculate_terrain_cost(cost_surface, target_type)
        
        # Base total cost on terrain cost
        cost_surface['total_cost'] = cost_surface[cost_col].copy()
        
        # Add tactical factors if requested
        if include_tactical:
            # Apply concealment inverse as a cost factor (low concealment = higher cost)
            if 'concealment' in cost_surface.columns:
                concealment_factor = 1 - cost_surface['concealment']
                # Scale the impact based on target type
                if target_type == 'infantry':
                    weight = 0.3  # Infantry values concealment highly
                elif target_type == 'vehicle':
                    weight = 0.1  # Vehicles value mobility over concealment
                else:
                    weight = 0.2  # Default value
                
                # Add weighted concealment factor to total cost
                cost_surface['total_cost'] += concealment_factor * weight
            
            # Apply distance to roads as a cost factor for vehicles
            if 'dist_to_road' in cost_surface.columns and target_type == 'vehicle':
                # Normalize road distance to 0-1 range
                max_road_dist = np.percentile(cost_surface['dist_to_road'].replace(np.inf, np.nan).dropna(), 95)
                road_factor = np.minimum(cost_surface['dist_to_road'] / max_road_dist, 1.0)
                
                # Add weighted road factor to total cost
                cost_surface['total_cost'] += road_factor * 0.2
        
        # Normalize total cost to 0-1 range
        min_cost = cost_surface['total_cost'].min()
        max_cost = cost_surface['total_cost'].max()
        
        if max_cost > min_cost:
            cost_surface['total_cost'] = (cost_surface['total_cost'] - min_cost) / (max_cost - min_cost)
        
        # Cache the cost surface if caching is enabled
        if self.config['cache_terrain']:
            try:
                cost_surface.to_pickle(cache_file)
                if self.verbose:
                    print(f"Cached cost surface to {cache_file}")
            except:
                if self.verbose:
                    print(f"Failed to cache cost surface")
        
        return cost_surface
    
    def identify_key_terrain(self, blue_forces_df=None, enemy_approach_direction=0):
        """
        Identify key terrain features based on elevation, observation, and mobility.
        
        Args:
            blue_forces_df (pd.DataFrame): Blue forces data for tactical considerations
            enemy_approach_direction (float): Expected enemy approach direction in degrees
            
        Returns:
            pd.DataFrame: Terrain grid with key terrain scores
        """
        if self.terrain_grid is None:
            if self.verbose:
                print("Error: Terrain grid not created yet")
                print("Call create_terrain_grid() first")
            return None
        
        # Check for cached key terrain analysis
        cache_key = f"key_terrain_dir{enemy_approach_direction}"
        if blue_forces_df is not None:
            cache_key += f"_blue{len(blue_forces_df)}"
        
        cache_file = os.path.join(self.config['cache_dir'], f'{cache_key}.pkl')
        
        if self.config['cache_terrain'] and os.path.exists(cache_file):
            try:
                if self.verbose:
                    print(f"Loading cached key terrain analysis from {cache_file}")
                
                return pd.read_pickle(cache_file)
            except:
                if self.verbose:
                    print("Failed to load cached key terrain analysis, recreating")
        
        if self.verbose:
            print("Identifying key terrain...")
        
        # Create a copy of the terrain grid
        key_terrain = self.terrain_grid.copy()
        
        # 1. Elevation Analysis
        if 'elevation' in key_terrain.columns:
            # Calculate local elevation prominence
            try:
                # Reshape to 2D grid
                elev_reshape = key_terrain['elevation'].values.reshape(self.grid_resolution, self.grid_resolution)
                
                # Apply a max filter to find local maxima
                from scipy.ndimage import maximum_filter, minimum_filter
                
                neighborhood_size = max(3, self.grid_resolution // 20)  # Adaptive size
                local_max = maximum_filter(elev_reshape, size=neighborhood_size)
                maxima = (elev_reshape == local_max)
                
                # Remove maxima at the edges
                maxima[:neighborhood_size,:] = False
                maxima[-neighborhood_size:,:] = False
                maxima[:,:neighborhood_size] = False
                maxima[:,-neighborhood_size:] = False
                
                # Calculate prominence (height difference to nearest higher ground)
                # For each maximum, calculate prominence
                prominence = np.zeros_like(elev_reshape)
                
                # Find coordinates of local maxima
                maxima_coords = np.where(maxima)
                
                for i, j in zip(*maxima_coords):
                    # Get elevation of this maximum
                    max_elev = elev_reshape[i, j]
                    
                    # Define a search region
                    search_size = min(neighborhood_size * 2, self.grid_resolution // 5)
                    
                    # Get bounds for search region
                    i_min = max(0, i - search_size)
                    i_max = min(elev_reshape.shape[0] - 1, i + search_size)
                    j_min = max(0, j - search_size)
                    j_max = min(elev_reshape.shape[1] - 1, j + search_size)
                    
                    # Extract search region
                    search_region = elev_reshape[i_min:i_max+1, j_min:j_max+1]
                    
                    # Find minimum elevation in search region
                    min_elev = np.min(search_region)
                    
                    # Calculate prominence
                    prom = max_elev - min_elev
                    
                    # Store prominence value
                    prominence[i, j] = prom
                
                # Reshape back to 1D and add to dataframe
                key_terrain['elevation_prominence'] = prominence.flatten()
                
                # Normalize prominence to 0-1 range
                max_prom = key_terrain['elevation_prominence'].max()
                if max_prom > 0:
                    key_terrain['elevation_score'] = key_terrain['elevation_prominence'] / max_prom
                else:
                    key_terrain['elevation_score'] = 0
            except:
                # Fall back to simple elevation percentile if reshaping fails
                key_terrain['elevation_score'] = key_terrain['elevation'].rank(pct=True)
        else:
            # No elevation data
            key_terrain['elevation_score'] = 0.5  # Neutral score
        
        # 2. Observation Analysis
        if 'elevation' in key_terrain.columns:
            # Calculate viewshed/observation score
            try:
                # Reshape to 2D grid
                elev_reshape = key_terrain['elevation'].values.reshape(self.grid_resolution, self.grid_resolution)
                
                # Calculate observation score based on viewshed analysis
                # This is computationally intensive, so we'll use a sampling approach
                
                # Sample points for viewshed calculation
                sample_interval = max(1, self.grid_resolution // 20)  # Adaptive sample interval
                i_samples = range(0, self.grid_resolution, sample_interval)
                j_samples = range(0, self.grid_resolution, sample_interval)
                
                # Calculate cell size in meters
                lat_range = key_terrain['latitude'].max() - key_terrain['latitude'].min()
                lon_range = key_terrain['longitude'].max() - key_terrain['longitude'].min()
                center_lat = (key_terrain['latitude'].max() + key_terrain['latitude'].min()) / 2
                
                lat_cell_size = lat_range / (self.grid_resolution - 1) * 111320  # m/deg at equator
                lon_cell_size = lon_range / (self.grid_resolution - 1) * 111320 * np.cos(np.radians(center_lat))
                
                cell_size = (lat_cell_size + lon_cell_size) / 2
                
                # Maximum view distance in grid cells
                max_view_dist = min(
                    self.grid_resolution // 2,  # Don't exceed half the grid
                    int(self.config['observation_range'] / cell_size)  # Convert meters to cells
                )
                
                # Create observer height offset (assume 2m observer height)
                observer_offset = 2  # meters
                
                # Initialize observation score
                observation_score = np.zeros_like(elev_reshape)
                
                # For each sample point, calculate viewshed
                for i in i_samples:
                    for j in j_samples:
                        # Observer position and elevation
                        obs_elev = elev_reshape[i, j] + observer_offset
                        
                        # Define view area bounds
                        i_min = max(0, i - max_view_dist)
                        i_max = min(elev_reshape.shape[0] - 1, i + max_view_dist)
                        j_min = max(0, j - max_view_dist)
                        j_max = min(elev_reshape.shape[1] - 1, j + max_view_dist)
                        
                        # Create view area grid
                        view_area = np.zeros((i_max - i_min + 1, j_max - j_min + 1), dtype=bool)
                        
                        # For each point in view area, check if visible
                        for vi in range(i_min, i_max + 1):
                            for vj in range(j_min, j_max + 1):
                                # Skip observer position
                                if vi == i and vj == j:
                                    continue
                                
                                # Calculate distance in grid cells
                                dist = np.sqrt((vi - i)**2 + (vj - j)**2)
                                
                                # Skip if beyond maximum view distance
                                if dist > max_view_dist:
                                    continue
                                
                                # Target position and elevation
                                target_elev = elev_reshape[vi, vj]
                                
                                # Check line of sight using Bresenham's line algorithm
                                # Create a line of points between observer and target
                                line_points = self._bresenham_line(i, j, vi, vj)
                                
                                # Check each point along the line
                                visible = True
                                
                                for pi, pj in line_points[1:-1]:  # Skip first (observer) and last (target) points
                                    # Point elevation
                                    point_elev = elev_reshape[pi, pj]
                                    
                                    # Calculate sight line elevation at this point
                                    # Use linear interpolation based on distance
                                    point_dist = np.sqrt((pi - i)**2 + (pj - j)**2)
                                    target_dist = dist
                                    
                                    sight_line_elev = obs_elev + (target_elev - obs_elev) * (point_dist / target_dist)
                                    
                                    # Check if point blocks line of sight
                                    if point_elev > sight_line_elev:
                                        visible = False
                                        break
                                
                                # Mark as visible in view area
                                if visible:
                                    view_area[vi - i_min, vj - j_min] = True
                        
                        # Calculate visible area size
                        visible_count = np.sum(view_area)
                        
                        # Add to observation score
                        observation_score[i, j] = visible_count
                
                # Normalize observation score to 0-1 range
                max_score = np.max(observation_score)
                if max_score > 0:
                    observation_score = observation_score / max_score
                
                # Reshape back to 1D and add to dataframe
                key_terrain['observation_score'] = observation_score.flatten()
            except:
                # Fall back to simpler method based on elevation
                if 'elevation_score' in key_terrain.columns:
                    key_terrain['observation_score'] = key_terrain['elevation_score'].copy()
                else:
                    key_terrain['observation_score'] = 0.5  # Neutral score
        else:
            # No elevation data
            key_terrain['observation_score'] = 0.5  # Neutral score
        
        # 3. Mobility Analysis
        if 'total_cost' in key_terrain.columns:
            # Inverse of cost is mobility (higher cost = lower mobility)
            key_terrain['mobility_score'] = 1 - key_terrain['total_cost']
        else:
            # No cost data, try using land use
            if 'land_use_type' in key_terrain.columns:
                # Calculate average mobility score across all target types
                mob_scores = []
                
                for target_type in self.config['mobility_classifications'].keys():
                    mob_col = f'{target_type}_cost'
                    if mob_col in key_terrain.columns:
                        mob_scores.append(1 - key_terrain[mob_col])
                
                if mob_scores:
                    key_terrain['mobility_score'] = np.mean(mob_scores, axis=0)
                else:
                    key_terrain['mobility_score'] = 0.5  # Neutral score
            else:
                key_terrain['mobility_score'] = 0.5  # Neutral score
        
        # 4. Tactical Considerations
        # Add concealment score
        if 'concealment' in key_terrain.columns:
            key_terrain['concealment_score'] = key_terrain['concealment'].copy()
        else:
            key_terrain['concealment_score'] = 0.5  # Neutral score
        
        # Consider enemy approach direction if specified
        if enemy_approach_direction is not None:
            # Calculate advantage based on aspect relative to enemy approach
            if 'aspect' in key_terrain.columns:
                # Convert to radians
                aspect_rad = np.radians(key_terrain['aspect'])
                enemy_rad = np.radians(enemy_approach_direction)
                
                # Calculate angular difference (0-180 degrees)
                diff_rad = np.abs(aspect_rad - enemy_rad)
                diff_rad = np.minimum(diff_rad, 2 * np.pi - diff_rad)
                diff_deg = np.degrees(diff_rad)
                
                # Normalize to 0-1 (1 = facing enemy, 0 = facing away)
                direction_score = 1 - (diff_deg / 180)
                
                # Add to key terrain assessment
                key_terrain['direction_score'] = direction_score
                
                # Factor into observation score (better observation facing enemy)
                key_terrain['observation_score'] = 0.7 * key_terrain['observation_score'] + 0.3 * direction_score
        
        # Consider blue forces positions if provided
        if blue_forces_df is not None and not blue_forces_df.empty:
            # Calculate distance to blue forces
            blue_tree = cKDTree(blue_forces_df[['latitude', 'longitude']].values)
            
            # Find nearest blue force for each terrain point
            distances, _ = blue_tree.query(key_terrain[['latitude', 'longitude']].values, k=1)
            
            # Convert to kilometers
            center_lat = (key_terrain['latitude'].min() + key_terrain['latitude'].max()) / 2
            km_per_degree = 111.32  # at equator
            km_per_degree_lon = km_per_degree * np.cos(np.radians(center_lat))
            
            # Calculate average km per degree for this region
            km_per_degree_avg = (km_per_degree + km_per_degree_lon) / 2
            
            # Convert to kilometers
            distances_km = distances * km_per_degree_avg
            
            # Normalize to 0-1 range (closer to blue forces = higher score)
            max_dist = min(20, np.percentile(distances_km, 95))  # Cap at 20km
            proximity_score = 1 - np.minimum(distances_km / max_dist, 1.0)
            
            # Add to key terrain assessment
            key_terrain['blue_proximity_score'] = proximity_score
            
            # Factor into overall assessment
            key_terrain['elevation_score'] = 0.8 * key_terrain['elevation_score'] + 0.2 * proximity_score
            key_terrain['observation_score'] = 0.8 * key_terrain['observation_score'] + 0.2 * proximity_score
        
        # 5. Calculate overall key terrain score
        # Get importance factors
        factors = self.config['key_terrain_factors']
        
        # Calculate weighted score
        key_terrain['key_terrain_score'] = (
            factors['elevation_importance'] * key_terrain['elevation_score'] +
            factors['observation_importance'] * key_terrain['observation_score'] +
            factors['mobility_importance'] * key_terrain['mobility_score'] +
            factors['concealment_importance'] * key_terrain['concealment_score']
        )
        
        # Identify top key terrain features
        # Find local maxima in key terrain score
        try:
            # Reshape to 2D grid
            score_reshape = key_terrain['key_terrain_score'].values.reshape(self.grid_resolution, self.grid_resolution)
            
            # Apply a max filter to find local maxima
            neighborhood_size = max(3, self.grid_resolution // 30)  # Adaptive size
            local_max = maximum_filter(score_reshape, size=neighborhood_size)
            maxima = (score_reshape == local_max) & (score_reshape > np.percentile(score_reshape, 80))
            
            # Label connected components
            from scipy.ndimage import label as ndlabel
            labeled_maxima, num_features = ndlabel(maxima)
            
            # Add feature label to key terrain
            key_terrain['key_feature_id'] = labeled_maxima.flatten()
            
            if self.verbose:
                print(f"Identified {num_features} key terrain features")
        except:
            # Skip feature labeling if reshaping fails
            pass
        
        # Cache the key terrain analysis if caching is enabled
        if self.config['cache_terrain']:
            try:
                key_terrain.to_pickle(cache_file)
                if self.verbose:
                    print(f"Cached key terrain analysis to {cache_file}")
            except:
                if self.verbose:
                    print(f"Failed to cache key terrain analysis")
        
        return key_terrain
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Implementation of Bresenham's line algorithm.
        
        Args:
            x0, y0: Start point coordinates
            x1, y1: End point coordinates
            
        Returns:
            list: List of points [(x,y), ...] along the line
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def identify_mobility_corridors(self, target_type='vehicle', max_cost_threshold=0.6):
        """
        Identify mobility corridors for the specified target type.
        
        Args:
            target_type (str): Target type (vehicle, infantry, artillery, command)
            max_cost_threshold (float): Maximum cost for mobility corridor
            
        Returns:
            dict: Mobility corridors information
        """
        if self.terrain_grid is None:
            if self.verbose:
                print("Error: Terrain grid not created yet")
                print("Call create_terrain_grid() first")
            return None
        
        # Check for cached mobility corridors
        cache_file = os.path.join(self.config['cache_dir'], f'mobility_corridors_{target_type}.pkl')
        
        if self.config['cache_terrain'] and os.path.exists(cache_file):
            try:
                if self.verbose:
                    print(f"Loading cached mobility corridors for {target_type} from {cache_file}")
                
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                if self.verbose:
                    print("Failed to load cached mobility corridors, recreating")
        
        if self.verbose:
            print(f"Identifying mobility corridors for {target_type}")
        
        # Get or create cost surface
        cost_col = f'{target_type}_cost'
        if cost_col not in self.terrain_grid.columns:
            cost_surface = self.create_cost_surface(target_type)
        else:
            cost_surface = self.terrain_grid.copy()
        
        # Reshape to 2D grid
        try:
            cost_reshape = cost_surface['total_cost'].values.reshape(self.grid_resolution, self.grid_resolution)
        except:
            if self.verbose:
                print("Error: Could not reshape cost surface to grid")
            return None
        
        # Identify low-cost areas (potential mobility corridors)
        low_cost_mask = cost_reshape < max_cost_threshold
        
        # Dilate the mask to connect nearby areas
        from scipy.ndimage import binary_dilation
        
        dilated_mask = binary_dilation(low_cost_mask, iterations=2)
        
        # Label connected components
        from scipy.ndimage import label as ndlabel
        
        labeled_corridors, num_corridors = ndlabel(dilated_mask)
        
        if self.verbose:
            print(f"Found {num_corridors} potential mobility corridors")
        
        # Calculate properties of each corridor
        corridor_properties = []
        
        for corridor_id in range(1, num_corridors + 1):
            # Get corridor mask
            corridor_mask = labeled_corridors == corridor_id
            
            # Calculate size (area)
            area = np.sum(corridor_mask)
            
            # Calculate centroid
            iy, ix = np.where(corridor_mask)
            centroid_x = np.mean(ix)
            centroid_y = np.mean(iy)
            
            # Convert to lat/lon
            lat_vals = np.linspace(self.lat_min, self.lat_max, self.grid_resolution)
            lon_vals = np.linspace(self.lon_min, self.lon_max, self.grid_resolution)
            
            centroid_lat = lat_vals[int(round(centroid_y))]
            centroid_lon = lon_vals[int(round(centroid_x))]
            
            # Calculate orientation (principal axis)
            if area > 10:  # Only for corridors of sufficient size
                try:
                    # Use PCA to find principal axis
                    coords = np.column_stack([iy, ix])
                    
                    # Center the coordinates
                    coords_centered = coords - [centroid_y, centroid_x]
                    
                    # Calculate covariance matrix
                    cov_matrix = np.cov(coords_centered.T)
                    
                    # Get eigenvectors and eigenvalues
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    
                    # Sort by eigenvalue in descending order
                    sort_indices = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[sort_indices]
                    eigenvectors = eigenvectors[:, sort_indices]
                    
                    # Principal axis is the first eigenvector
                    principal_axis = eigenvectors[:, 0]
                    
                    # Calculate orientation in degrees
                    orientation = np.degrees(np.arctan2(principal_axis[1], principal_axis[0])) % 180
                    
                    # Calculate corridor shape (ratio of eigenvalues)
                    shape_ratio = eigenvalues[0] / max(eigenvalues[1], 1e-10)
                except:
                    orientation = 0
                    shape_ratio = 1
            else:
                orientation = 0
                shape_ratio = 1
            
            # Calculate average cost
            avg_cost = np.mean(cost_reshape[corridor_mask])
            
            # Calculate width and length
            if shape_ratio > 1:
                # Elongated corridor
                length = np.sqrt(area * shape_ratio)
                width = area / length
            else:
                # Roughly circular or irregular
                length = np.sqrt(area)
                width = length
            
            # Convert to km
            center_lat = (cost_surface['latitude'].min() + cost_surface['latitude'].max()) / 2
            km_per_degree = 111.32  # at equator
            km_per_degree_lon = km_per_degree * np.cos(np.radians(center_lat))
            
            # Calculate average km per degree for this region
            km_per_degree_avg = (km_per_degree + km_per_degree_lon) / 2
            
            # Calculate grid cell size in km
            cell_size_km = (self.lat_max - self.lat_min) / self.grid_resolution * km_per_degree_avg
            
            # Convert to km
            length_km = length * cell_size_km
            width_km = width * cell_size_km
            
            # Calculate coordinates of corridor boundary
            boundary_y, boundary_x = np.where(
                corridor_mask & ~binary_dilation(corridor_mask, iterations=1)
            )
            
            # Convert to lat/lon
            boundary_points = []
            for by, bx in zip(boundary_y, boundary_x):
                boundary_lat = lat_vals[by]
                boundary_lon = lon_vals[bx]
                boundary_points.append((boundary_lat, boundary_lon))
            
            # Store corridor properties
            corridor_props = {
                'corridor_id': corridor_id,
                'area': area,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'orientation': orientation,
                'shape_ratio': shape_ratio,
                'avg_cost': avg_cost,
                'length_km': length_km,
                'width_km': width_km,
                'boundary': boundary_points
            }
            
            corridor_properties.append(corridor_props)
        
        # Sort corridors by area (descending)
        corridor_properties.sort(key=lambda x: x['area'], reverse=True)
        
        # Filter out very small corridors
        min_area = max(10, self.grid_resolution // 50)
        corridor_properties = [c for c in corridor_properties if c['area'] > min_area]
        
        # Create result
        mobility_corridors = {
            'target_type': target_type,
            'num_corridors': len(corridor_properties),
            'corridors': corridor_properties,
            'cost_threshold': max_cost_threshold
        }
        
        # Cache the mobility corridors if caching is enabled
        if self.config['cache_terrain']:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(mobility_corridors, f)
                
                if self.verbose:
                    print(f"Cached mobility corridors to {cache_file}")
            except:
                if self.verbose:
                    print(f"Failed to cache mobility corridors")
        
        return mobility_corridors
    
    def create_mobility_network(self, target_type='vehicle', node_spacing=10, max_edge_cost=0.8):
        """
        Create a mobility network for pathfinding and movement analysis.
        
        Args:
            target_type (str): Target type (vehicle, infantry, artillery, command)
            node_spacing (int): Spacing between network nodes in grid cells
            max_edge_cost (float): Maximum cost for network edges
            
        Returns:
            networkx.Graph: Mobility network for pathfinding
        """
        if self.terrain_grid is None:
            if self.verbose:
                print("Error: Terrain grid not created yet")
                print("Call create_terrain_grid() first")
            return None
        
        # Check for cached mobility network
        cache_file = os.path.join(self.config['cache_dir'], f'mobility_network_{target_type}_{node_spacing}.pkl')
        
        if self.config['cache_terrain'] and os.path.exists(cache_file):
            try:
                if self.verbose:
                    print(f"Loading cached mobility network for {target_type} from {cache_file}")
                
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                if self.verbose:
                    print("Failed to load cached mobility network, recreating")
        
        if self.verbose:
            print(f"Creating mobility network for {target_type}")
        
        # Get or create cost surface
        cost_col = f'{target_type}_cost'
        if 'total_cost' not in self.terrain_grid.columns:
            cost_surface = self.create_cost_surface(target_type)
        else:
            cost_surface = self.terrain_grid.copy()
        
        # Reshape to 2D grid
        try:
            cost_reshape = cost_surface['total_cost'].values.reshape(self.grid_resolution, self.grid_resolution)
        except:
            if self.verbose:
                print("Error: Could not reshape cost surface to grid")
            return None
        
        # Create a graph
        G = nx.Graph()
        
        # Create nodes at regular intervals
        for i in range(0, self.grid_resolution, node_spacing):
            for j in range(0, self.grid_resolution, node_spacing):
                # Get node position
                lat_idx = min(i, self.grid_resolution - 1)
                lon_idx = min(j, self.grid_resolution - 1)
                
                # Convert to lat/lon
                lat_vals = np.linspace(self.lat_min, self.lat_max, self.grid_resolution)
                lon_vals = np.linspace(self.lon_min, self.lon_max, self.grid_resolution)
                
                lat = lat_vals[lat_idx]
                lon = lon_vals[lon_idx]
                
                # Get cost at this position
                cost = cost_reshape[lat_idx, lon_idx]
                
                # Skip if cost is very high (impassable)
                if cost > 0.9:
                    continue
                
                # Create node
                node_id = f"{lat_idx}_{lon_idx}"
                G.add_node(node_id, latitude=lat, longitude=lon, cost=cost, grid_i=lat_idx, grid_j=lon_idx)
        
        # Create edges between neighboring nodes
        nodes = list(G.nodes(data=True))
        
        # Calculate distance threshold (diagonal of node spacing)
        dist_threshold = node_spacing * np.sqrt(2) * 1.1  # Allow some margin
        
        # Create KD-tree for efficient neighbor finding
        node_positions = np.array([[node[1]['grid_i'], node[1]['grid_j']] for node in nodes])
        node_tree = cKDTree(node_positions)
        
        # Find and add edges
        for i, (node_id, node_data) in enumerate(nodes):
            # Find neighboring nodes
            neighbors = node_tree.query_ball_point(
                [node_data['grid_i'], node_data['grid_j']],
                dist_threshold
            )
            
            for j in neighbors:
                # Skip self
                if i == j:
                    continue
                
                neighbor_id = nodes[j][0]
                neighbor_data = nodes[j][1]
                
                # Calculate edge cost (average of node costs)
                edge_cost = (node_data['cost'] + neighbor_data['cost']) / 2
                
                # Skip if edge cost is too high
                if edge_cost > max_edge_cost:
                    continue
                
                # Calculate distance
                dist = np.sqrt(
                    (node_data['grid_i'] - neighbor_data['grid_i'])**2 +
                    (node_data['grid_j'] - neighbor_data['grid_j'])**2
                )
                
                # Create weighted edge
                G.add_edge(
                    node_id,
                    neighbor_id,
                    cost=edge_cost,
                    weight=edge_cost * dist,  # Weight by cost and distance
                    distance=dist
                )
        
        if self.verbose:
            print(f"Created mobility network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Cache the mobility network if caching is enabled
        if self.config['cache_terrain']:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(G, f)
                
                if self.verbose:
                    print(f"Cached mobility network to {cache_file}")
            except:
                if self.verbose:
                    print(f"Failed to cache mobility network")
        
        return G
    
    def find_optimal_path(self, start_coords, end_coords, target_type='vehicle', mobility_network=None):
        """
        Find optimal path between two points for the specified target type.
        
        Args:
            start_coords (tuple): Start coordinates (latitude, longitude)
            end_coords (tuple): End coordinates (latitude, longitude)
            target_type (str): Target type (vehicle, infantry, artillery, command)
            mobility_network (networkx.Graph): Pre-created mobility network (optional)
            
        Returns:
            dict: Path information including coordinates, cost, and distance
        """
        # Get mobility network
        if mobility_network is None:
            mobility_network = self.create_mobility_network(target_type)
        
        if mobility_network is None or mobility_network.number_of_nodes() == 0:
            if self.verbose:
                print("Error: Could not create mobility network")
            return None
        
        # Find nearest nodes to start and end points
        nodes = list(mobility_network.nodes(data=True))
        node_positions = np.array([[node[1]['latitude'], node[1]['longitude']] for node in nodes])
        node_ids = [node[0] for node in nodes]
        
        node_tree = cKDTree(node_positions)
        
        # Find nearest nodes
        start_dist, start_idx = node_tree.query(start_coords, k=1)
        end_dist, end_idx = node_tree.query(end_coords, k=1)
        
        start_node = node_ids[start_idx]
        end_node = node_ids[end_idx]
        
        # Find shortest path using Dijkstra's algorithm
        try:
            path = nx.shortest_path(mobility_network, start_node, end_node, weight='weight')
        except nx.NetworkXNoPath:
            if self.verbose:
                print(f"No path found between ({start_coords}) and ({end_coords})")
            return None
        
        # Extract path coordinates and properties
        path_coords = []
        total_cost = 0
        total_distance = 0
        
        for i in range(len(path)):
            node_id = path[i]
            node_data = mobility_network.nodes[node_id]
            
            path_coords.append((node_data['latitude'], node_data['longitude']))
            
            # Calculate segment cost and distance
            if i > 0:
                prev_node = path[i-1]
                edge_data = mobility_network.edges[prev_node, node_id]
                total_cost += edge_data['cost']
                total_distance += edge_data['distance']
        
        # Convert distance to kilometers
        center_lat = (self.lat_min + self.lat_max) / 2
        km_per_degree = 111.32  # at equator
        km_per_degree_lon = km_per_degree * np.cos(np.radians(center_lat))
        
        # Calculate average km per degree for this region
        km_per_degree_avg = (km_per_degree + km_per_degree_lon) / 2
        
        # Calculate grid cell size in km
        cell_size_km = (self.lat_max - self.lat_min) / self.grid_resolution * km_per_degree_avg
        
        # Convert to km
        total_distance_km = total_distance * cell_size_km
        
        # Create result
        path_result = {
            'target_type': target_type,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'path_coords': path_coords,
            'total_cost': total_cost,
            'avg_cost': total_cost / max(1, len(path) - 1),
            'total_distance_km': total_distance_km,
            'path_nodes': path
        }
        
        return path_result
    
    def analyze_line_of_sight(self, observer_coords, target_coords, observer_height=2.0, target_height=0.0):
        """
        Analyze line of sight between observer and target.
        
        Args:
            observer_coords (tuple): Observer coordinates (latitude, longitude)
            target_coords (tuple): Target coordinates (latitude, longitude)
            observer_height (float): Additional height of observer (meters)
            target_height (float): Additional height of target (meters)
            
        Returns:
            dict: Line of sight analysis results
        """
        if self.terrain_grid is None or 'elevation' not in self.terrain_grid.columns:
            if self.verbose:
                print("Error: Terrain grid with elevation data required")
            return None
        
        # Check if we have a spatial index
        if self.spatial_index is None:
            self._create_spatial_index()
        
        # Get elevations at observer and target
        distances, indices = self.spatial_index.query([observer_coords, target_coords], k=1)
        
        observer_elev = self.terrain_grid.iloc[indices[0]]['elevation'] + observer_height
        target_elev = self.terrain_grid.iloc[indices[1]]['elevation'] + target_height
        
        # Calculate distance between points
        from scipy.spatial.distance import great_circle
        
        dist_km = great_circle(observer_coords, target_coords).kilometers
        dist_m = dist_km * 1000  # Convert to meters
        
        # Create a line between observer and target
        # Number of sample points (at least 10, more for longer distances)
        num_points = max(10, int(dist_km * 10))
        
        # Create sample points along the line
        lat_steps = np.linspace(observer_coords[0], target_coords[0], num_points)
        lon_steps = np.linspace(observer_coords[1], target_coords[1], num_points)
        
        line_points = list(zip(lat_steps, lon_steps))
        
        # Get elevations along the line
        line_distances, line_indices = self.spatial_index.query(line_points, k=1)
        line_elevations = [self.terrain_grid.iloc[idx]['elevation'] for idx in line_indices]
        
        # Calculate sight line elevations
        # Linear interpolation from observer to target
        sight_line = []
        
        for i in range(num_points):
            # Calculate fraction of distance
            fraction = i / (num_points - 1)
            
            # Calculate sight line elevation at this point
            sight_elev = observer_elev + fraction * (target_elev - observer_elev)
            sight_line.append(sight_elev)
        
        # Check if any terrain points block the sight line
        blocked = False
        blocking_point = None
        blocking_distance = 0
        
        for i in range(1, num_points - 1):  # Skip observer and target points
            if line_elevations[i] > sight_line[i]:
                blocked = True
                blocking_point = line_points[i]
                
                # Calculate distance to blocking point
                blocking_distance = great_circle(observer_coords, blocking_point).kilometers * 1000  # m
                break
        
        # Calculate maximum sight line clearance (or minimum interference)
        clearances = [sight_line[i] - line_elevations[i] for i in range(num_points)]
        min_clearance = min(clearances)
        
        # Calculate point of closest approach (minimum clearance)
        closest_idx = np.argmin(clearances)
        closest_point = line_points[closest_idx]
        closest_distance = great_circle(observer_coords, closest_point).kilometers * 1000  # m
        
        # Create result
        los_result = {
            'visible': not blocked,
            'distance_m': dist_m,
            'observer_elevation_m': observer_elev,
            'target_elevation_m': target_elev,
            'min_clearance_m': min_clearance,
            'blocked': blocked
        }
        
        if blocked:
            los_result.update({
                'blocking_point': blocking_point,
                'blocking_distance_m': blocking_distance
            })
        
        los_result.update({
            'closest_approach_point': closest_point,
            'closest_approach_distance_m': closest_distance,
            'line_points': line_points,
            'line_elevations': line_elevations,
            'sight_line_elevations': sight_line
        })
        
        return los_result
    
    def analyze_visibility_area(self, observer_coords, max_range_km=5.0, observer_height=2.0, sampling=10):
        """
        Analyze visibility area from an observer position.
        
        Args:
            observer_coords (tuple): Observer coordinates (latitude, longitude)
            max_range_km (float): Maximum visibility range in kilometers
            observer_height (float): Additional height of observer (meters)
            sampling (int): Sampling density (higher = more accurate, slower)
            
        Returns:
            dict: Visibility analysis results
        """
        if self.terrain_grid is None or 'elevation' not in self.terrain_grid.columns:
            if self.verbose:
                print("Error: Terrain grid with elevation data required")
            return None
        
        # Create a sampling grid
        # Calculate angular resolution (approximately)
        angular_res = sampling / max_range_km  # degrees per km
        
        # Create a circular sampling pattern
        # Number of angular samples
        num_angles = int(360 * angular_res)
        
        # Create a list of angles (in degrees)
        angles = np.linspace(0, 360, num_angles, endpoint=False)
        
        # Create a list of distances
        num_distances = int(max_range_km * sampling)
        distances = np.linspace(0.1, max_range_km, num_distances)  # Start slightly away from observer
        
        # Get observer elevation
        if self.spatial_index is None:
            self._create_spatial_index()
        
        distances_obs, indices_obs = self.spatial_index.query([observer_coords], k=1)
        observer_elev = self.terrain_grid.iloc[indices_obs[0]]['elevation'] + observer_height
        
        # Create result structures
        visible_points = []
        non_visible_points = []
        visibility_distances = {}  # Maximum visibility distance by angle
        
        # For each angle
        for angle in angles:
            # Convert to radians
            angle_rad = np.radians(angle)
            
            # Maximum visible distance for this angle
            max_visible_dist = max_range_km
            
            # Check each distance along this angle
            for dist in distances:
                # Calculate target point
                # Convert polar to Cartesian
                # Note: We use a simple planar approximation which is good enough for small areas
                dx = dist * np.sin(angle_rad)
                dy = dist * np.cos(angle_rad)
                
                # Convert to lat/lon offset
                # 111.32 km per degree latitude at equator
                # 111.32 * cos(lat) km per degree longitude
                lat_offset = dy / 111.32
                lon_offset = dx / (111.32 * np.cos(np.radians(observer_coords[0])))
                
                # Calculate target coordinates
                target_lat = observer_coords[0] + lat_offset
                target_lon = observer_coords[1] + lon_offset
                
                # Check if within terrain bounds
                if (self.lat_min <= target_lat <= self.lat_max and
                    self.lon_min <= target_lon <= self.lon_max):
                    
                    # Check line of sight
                    los = self.analyze_line_of_sight(
                        observer_coords, 
                        (target_lat, target_lon),
                        observer_height,
                        0
                    )
                    
                    if los['visible']:
                        visible_points.append((target_lat, target_lon))
                    else:
                        non_visible_points.append((target_lat, target_lon))
                        # Update maximum visible distance
                        if dist < max_visible_dist:
                            max_visible_dist = los['blocking_distance_m'] / 1000  # Convert to km
                            break
            
            # Store maximum visibility distance for this angle
            visibility_distances[angle] = max_visible_dist
        
        # Calculate visible area
        # Create a convex hull of visible points
        if len(visible_points) >= 3:
            try:
                from scipy.spatial import ConvexHull
                
                # Create convex hull
                visible_points_array = np.array(visible_points)
                hull = ConvexHull(visible_points_array)
                
                # Calculate area in square kilometers
                hull_points = visible_points_array[hull.vertices]
                
                # Approximate area calculation
                lat_min = np.min(hull_points[:, 0])
                lat_max = np.max(hull_points[:, 0])
                lon_min = np.min(hull_points[:, 1])
                lon_max = np.max(hull_points[:, 1])
                
                # Calculate area in square kilometers
                # 111.32 km per degree latitude at equator
                # 111.32 * cos(lat) km per degree longitude
                center_lat = (lat_min + lat_max) / 2
                km_per_degree_lat = 111.32
                km_per_degree_lon = 111.32 * np.cos(np.radians(center_lat))
                
                # Approximate area as percentage of bounding rectangle
                rect_area = (lat_max - lat_min) * km_per_degree_lat * (lon_max - lon_min) * km_per_degree_lon
                visible_area = hull.volume * rect_area / ((lat_max - lat_min) * (lon_max - lon_min))
                
                # Get hull perimeter
                hull_perimeter = 0
                for i in range(len(hull.vertices)):
                    idx1 = hull.vertices[i]
                    idx2 = hull.vertices[(i + 1) % len(hull.vertices)]
                    
                    p1 = visible_points_array[idx1]
                    p2 = visible_points_array[idx2]
                    
                    # Calculate distance in kilometers
                    segment_dist = great_circle(p1, p2).kilometers # type: ignore
                    
                    hull_perimeter += segment_dist
            except:
                visible_area = 0
                hull_perimeter = 0
                hull_points = []
        else:
            visible_area = 0
            hull_perimeter = 0
            hull_points = []
        
        # Create result
        visibility_result = {
            'observer_coords': observer_coords,
            'observer_elevation_m': observer_elev,
            'max_range_km': max_range_km,
            'visible_points': visible_points,
            'non_visible_points': non_visible_points,
            'visibility_distances': visibility_distances,
            'visible_area_km2': visible_area,
            'visible_perimeter_km': hull_perimeter,
            'visible_hull_points': hull_points if len(hull_points) > 0 else []
        }
        
        return visibility_result
    
    def find_optimal_observation_points(self, area_coords, num_points=3, max_range_km=5.0, min_elevation_percentile=70):
        """
        Find optimal observation points to cover an area.
        
        Args:
            area_coords (list): List of (latitude, longitude) coordinates defining the area
            num_points (int): Number of observation points to find
            max_range_km (float): Maximum visibility range in kilometers
            min_elevation_percentile (float): Minimum elevation percentile for candidate points
            
        Returns:
            list: List of optimal observation points
        """
        if self.terrain_grid is None or 'elevation' not in self.terrain_grid.columns:
            if self.verbose:
                print("Error: Terrain grid with elevation data required")
            return None
        
        # Find points within or near the specified area
        # Calculate area centroid
        area_lat = [p[0] for p in area_coords]
        area_lon = [p[1] for p in area_coords]
        
        centroid_lat = np.mean(area_lat)
        centroid_lon = np.mean(area_lon)
        
        # Calculate search radius
        radius = max(
            great_circle((centroid_lat, centroid_lon), (min(area_lat), min(area_lon))).kilometers, # type: ignore
            great_circle((centroid_lat, centroid_lon), (max(area_lat), max(area_lon))).kilometers # type: ignore
        )
        
        # Add buffer
        search_radius = radius * 1.2
        
        # Find candidate points (high elevation points)
        # Calculate elevation threshold
        elev_threshold = np.percentile(self.terrain_grid['elevation'], min_elevation_percentile)
        
        # Find high points
        high_points = self.terrain_grid[self.terrain_grid['elevation'] >= elev_threshold]
        
        # Filter by distance to area centroid
        distances = [
            great_circle((p['latitude'], p['longitude']), (centroid_lat, centroid_lon)).kilometers # type: ignore
            for _, p in high_points.iterrows()
        ]
        
        mask = [d <= search_radius for d in distances]
        candidate_points = high_points.iloc[np.where(mask)[0]]
        
        if len(candidate_points) == 0:
            if self.verbose:
                print(f"No candidate points found with elevation >= {elev_threshold}m within {search_radius}km of area")
            
            # Fall back to using all points in the area
            candidate_points = self.terrain_grid.copy()
        
        # Analyze coverage of each candidate point
        point_coverage = []
        
        # Create a simplified sampling of the area for coverage calculation
        area_sampling = []
        
        # Use a grid pattern to sample the area
        lat_min, lat_max = min(area_lat), max(area_lat)
        lon_min, lon_max = min(area_lon), max(area_lon)
        
        # Calculate coverage sampling resolution
        # Use at most 100 points
        area_size = great_circle((lat_min, lon_min), (lat_max, lon_max)).kilometers # type: ignore
        sample_resolution = max(1, int(np.sqrt(area_size)))
        
        # Create sampling grid
        lat_samples = np.linspace(lat_min, lat_max, sample_resolution)
        lon_samples = np.linspace(lon_min, lon_max, sample_resolution)
        
        for lat in lat_samples:
            for lon in lon_samples:
                area_sampling.append((lat, lon))
        
        # Evaluate each candidate point
        for idx, point in candidate_points.iterrows():
            observer_coords = (point['latitude'], point['longitude'])
            
            # Quick estimate of visible coverage
            visible_count = 0
            
            for target_coords in area_sampling:
                # Check if target is within max range
                dist = great_circle(observer_coords, target_coords).kilometers # type: ignore
                
                if dist <= max_range_km:
                    # Check line of sight
                    los = self.analyze_line_of_sight(
                        observer_coords,
                        target_coords,
                        observer_height=2.0,
                        target_height=0.0
                    )
                    
                    if los['visible']:
                        visible_count += 1
            
            # Calculate coverage percentage
            coverage_pct = visible_count / len(area_sampling) * 100
            
            # Add to point coverage
            point_coverage.append({
                'coords': observer_coords,
                'elevation': point['elevation'],
                'coverage_pct': coverage_pct
            })
        
        # Sort by coverage
        point_coverage.sort(key=lambda x: x['coverage_pct'], reverse=True)
        
        # Greedily select best points with minimal overlap
        selected_points = []
        covered_targets = set()
        
        while len(selected_points) < num_points and point_coverage:
            # Get best remaining point
            best_point = point_coverage[0]
            point_coverage = point_coverage[1:]
            
            # Add to selected points
            selected_points.append(best_point)
            
            # Update covered targets
            observer_coords = best_point['coords']
            
            for i, target_coords in enumerate(area_sampling):
                # Check if target is within max range
                dist = great_circle(observer_coords, target_coords).kilometers # type: ignore
                
                if dist <= max_range_km:
                    # Check line of sight
                    los = self.analyze_line_of_sight(
                        observer_coords,
                        target_coords,
                        observer_height=2.0,
                        target_height=0.0
                    )
                    
                    if los['visible']:
                        covered_targets.add(i)
            
            # Recalculate coverage for remaining points
            if len(selected_points) < num_points:
                for i in range(len(point_coverage)):
                    observer_coords = point_coverage[i]['coords']
                    
                    # Count targets that aren't already covered
                    new_coverage = 0
                    
                    for j, target_coords in enumerate(area_sampling):
                        if j in covered_targets:
                            continue
                        
                        # Check if target is within max range
                        dist = great_circle(observer_coords, target_coords).kilometers # type: ignore
                        
                        if dist <= max_range_km:
                            # Check line of sight
                            los = self.analyze_line_of_sight(
                                observer_coords,
                                target_coords,
                                observer_height=2.0,
                                target_height=0.0
                            )
                            
                            if los['visible']:
                                new_coverage += 1
                    
                    # Update coverage percentage (based on uncovered targets)
                    uncovered_count = len(area_sampling) - len(covered_targets)
                    
                    if uncovered_count > 0:
                        point_coverage[i]['coverage_pct'] = new_coverage / uncovered_count * 100
                    else:
                        point_coverage[i]['coverage_pct'] = 0
                
                # Resort by coverage
                point_coverage.sort(key=lambda x: x['coverage_pct'], reverse=True)
        
        # Calculate total coverage
        total_coverage_pct = len(covered_targets) / len(area_sampling) * 100
        
        # Create result
        result = {
            'observation_points': selected_points,
            'total_coverage_pct': total_coverage_pct,
            'area_coords': area_coords,
            'area_centroid': (centroid_lat, centroid_lon),
            'max_range_km': max_range_km
        }
        
        return result
    
    def classify_terrain(self, area_coords=None):
        """
        Classify terrain characteristics for military analysis.
        
        Args:
            area_coords (list): List of (latitude, longitude) coordinates defining the area
            
        Returns:
            dict: Terrain classification
        """
        if self.terrain_grid is None:
            if self.verbose:
                print("Error: Terrain grid not created yet")
                print("Call create_terrain_grid() first")
            return None
        
        # Use the entire terrain grid if no area specified
        if area_coords is None:
            terrain_data = self.terrain_grid
        else:
            # Filter terrain grid to the specified area
            # Calculate area bounds
            area_lat = [p[0] for p in area_coords]
            area_lon = [p[1] for p in area_coords]
            
            lat_min, lat_max = min(area_lat), max(area_lat)
            lon_min, lon_max = min(area_lon), max(area_lon)
            
            # Filter by bounds
            mask = (
                (self.terrain_grid['latitude'] >= lat_min) &
                (self.terrain_grid['latitude'] <= lat_max) &
                (self.terrain_grid['longitude'] >= lon_min) &
                (self.terrain_grid['longitude'] <= lon_max)
            )
            
            terrain_data = self.terrain_grid[mask]
            
            if terrain_data.empty:
                if self.verbose:
                    print("No terrain data found in specified area")
                return None
        
        # Calculate terrain statistics
        result = {
            'area_size_km2': 0,
            'elevation_stats': {},
            'slope_stats': {},
            'land_use_distribution': {},
            'terrain_classification': {},
            'key_terrain_features': [],
            'mobility_assessment': {}
        }
        
        # Calculate area size
        if area_coords:
            # Calculate area of polygon
            from shapely.geometry import Polygon # type: ignore
            
            try:
                poly = Polygon(area_coords)
                
                # Approximate area in square kilometers
                lat_min, lat_max = min(area_lat), max(area_lat)
                lon_min, lon_max = min(area_lon), max(area_lon)
                
                # Calculate area in square kilometers
                # 111.32 km per degree latitude at equator
                # 111.32 * cos(lat) km per degree longitude
                center_lat = (lat_min + lat_max) / 2
                km_per_degree_lat = 111.32
                km_per_degree_lon = 111.32 * np.cos(np.radians(center_lat))
                
                # Calculate approximate area in square kilometers
                lon_extent = lon_max - lon_min
                lat_extent = lat_max - lat_min
                
                # Area of bounding rectangle
                rect_area = lat_extent * km_per_degree_lat * lon_extent * km_per_degree_lon
                
                # Scale by ratio of polygon area to rectangle area
                result['area_size_km2'] = poly.area / (lat_extent * lon_extent) * rect_area
            except:
                # Fall back to bounding rectangle area
                lat_min, lat_max = min(area_lat), max(area_lat)
                lon_min, lon_max = min(area_lon), max(area_lon)
                
                # Calculate area in square kilometers
                center_lat = (lat_min + lat_max) / 2
                km_per_degree_lat = 111.32
                km_per_degree_lon = 111.32 * np.cos(np.radians(center_lat))
                
                result['area_size_km2'] = (lat_max - lat_min) * km_per_degree_lat * (lon_max - lon_min) * km_per_degree_lon
        else:
            # Calculate area of entire terrain grid
            lat_min, lat_max = self.lat_min, self.lat_max
            lon_min, lon_max = self.lon_min, self.lon_max
            
            # Calculate area in square kilometers
            center_lat = (lat_min + lat_max) / 2
            km_per_degree_lat = 111.32
            km_per_degree_lon = 111.32 * np.cos(np.radians(center_lat))
            
            result['area_size_km2'] = (lat_max - lat_min) * km_per_degree_lat * (lon_max - lon_min) * km_per_degree_lon
        
        # Elevation statistics
        if 'elevation' in terrain_data.columns:
            elev_stats = terrain_data['elevation'].describe()
            
            result['elevation_stats'] = {
                'min_m': elev_stats['min'],
                'max_m': elev_stats['max'],
                'mean_m': elev_stats['mean'],
                'std_dev_m': elev_stats['std'],
                'range_m': elev_stats['max'] - elev_stats['min'],
                'median_m': terrain_data['elevation'].median(),
                'percentiles': {
                    '10': np.percentile(terrain_data['elevation'], 10),
                    '25': np.percentile(terrain_data['elevation'], 25),
                    '75': np.percentile(terrain_data['elevation'], 75),
                    '90': np.percentile(terrain_data['elevation'], 90)
                }
            }
            
            # Classify terrain by relief
            elev_range = result['elevation_stats']['range_m']
            
            if elev_range < 30:
                result['terrain_classification']['relief'] = 'flat'
            elif elev_range < 100:
                result['terrain_classification']['relief'] = 'gently rolling'
            elif elev_range < 300:
                result['terrain_classification']['relief'] = 'hilly'
            else:
                result['terrain_classification']['relief'] = 'mountainous'
        
        # Slope statistics
        if 'slope' in terrain_data.columns:
            slope_stats = terrain_data['slope'].describe()
            
            result['slope_stats'] = {
                'min_deg': slope_stats['min'],
                'max_deg': slope_stats['max'],
                'mean_deg': slope_stats['mean'],
                'std_dev_deg': slope_stats['std'],
                'median_deg': terrain_data['slope'].median(),
                'percentiles': {
                    '10': np.percentile(terrain_data['slope'], 10),
                    '25': np.percentile(terrain_data['slope'], 25),
                    '75': np.percentile(terrain_data['slope'], 75),
                    '90': np.percentile(terrain_data['slope'], 90)
                }
            }
            
            # Calculate percentage of terrain in each slope category
            slope_cats = {
                'level': (0, 3),
                'gentle': (3, 10),
                'moderate': (10, 20),
                'steep': (20, 30),
                'very_steep': (30, float('inf'))
            }
            
            slope_dist = {}
            
            for cat, (min_slope, max_slope) in slope_cats.items():
                count = ((terrain_data['slope'] >= min_slope) & (terrain_data['slope'] < max_slope)).sum()
                slope_dist[cat] = count / len(terrain_data) * 100
            
            result['slope_stats']['distribution'] = slope_dist
            
            # Classify terrain by trafficability
            if slope_dist['level'] + slope_dist['gentle'] > 70:
                result['terrain_classification']['trafficability'] = 'high'
            elif slope_dist['level'] + slope_dist['gentle'] > 40:
                result['terrain_classification']['trafficability'] = 'moderate'
            else:
                result['terrain_classification']['trafficability'] = 'low'
        
        # Land use distribution
        if 'land_use_type' in terrain_data.columns:
            land_use_counts = terrain_data['land_use_type'].value_counts(normalize=True) * 100
            result['land_use_distribution'] = land_use_counts.to_dict()
            
            # Classify terrain by cover and concealment
            forest_pct = land_use_counts.get('forest', 0)
            urban_pct = land_use_counts.get('urban', 0)
            light_veg_pct = land_use_counts.get('light_vegetation', 0)
            
            cover_concealment = forest_pct + urban_pct + 0.5 * light_veg_pct
            
            if cover_concealment > 70:
                result['terrain_classification']['cover'] = 'excellent'
            elif cover_concealment > 40:
                result['terrain_classification']['cover'] = 'good'
            elif cover_concealment > 20:
                result['terrain_classification']['cover'] = 'moderate'
            else:
                result['terrain_classification']['cover'] = 'poor'
            
            # Classify by dominant terrain type
            dominant_type = land_use_counts.idxmax()
            result['terrain_classification']['dominant_type'] = dominant_type
        
        # Assess mobility for different target types
        for target_type in self.config['mobility_classifications'].keys():
            cost_col = f'{target_type}_cost'
            
            if cost_col in terrain_data.columns:
                cost_stats = terrain_data[cost_col].describe()
                
                result['mobility_assessment'][target_type] = {
                    'min_cost': cost_stats['min'],
                    'max_cost': cost_stats['max'],
                    'mean_cost': cost_stats['mean'],
                    'median_cost': terrain_data[cost_col].median()
                }
                
                # Classify mobility
                mean_cost = cost_stats['mean']
                
                if mean_cost < 0.3:
                    mobility = 'high'
                elif mean_cost < 0.6:
                    mobility = 'moderate'
                else:
                    mobility = 'low'
                
                result['mobility_assessment'][target_type]['classification'] = mobility
        
        # Identify key terrain features
        # If we have key terrain analysis
        if 'key_terrain_score' in terrain_data.columns and 'key_feature_id' in terrain_data.columns:
            # Get key features
            feature_ids = terrain_data['key_feature_id'].unique()
            
            for feature_id in feature_ids:
                if feature_id == 0:
                    continue  # Skip areas not identified as key terrain
                
                # Get feature data
                feature_data = terrain_data[terrain_data['key_feature_id'] == feature_id]
                
                # Get most significant point
                max_score_idx = feature_data['key_terrain_score'].idxmax()
                max_point = feature_data.loc[max_score_idx]
                
                # Add to key terrain features
                feature = {
                    'feature_id': int(feature_id),
                    'coords': (float(max_point['latitude']), float(max_point['longitude'])),
                    'elevation_m': float(max_point['elevation']),
                    'score': float(max_point['key_terrain_score']),
                    'size': len(feature_data)
                }
                
                # Add observation score if available
                if 'observation_score' in max_point:
                    feature['observation_score'] = float(max_point['observation_score'])
                
                result['key_terrain_features'].append(feature)
            
            # Sort by score
            result['key_terrain_features'].sort(key=lambda x: x['score'], reverse=True)
        
        # Identify potential obstacles
        # Areas with high cost for vehicles
        if 'vehicle_cost' in terrain_data.columns:
            # Find contiguous regions of high cost
            try:
                # Reshape to 2D grid
                cost_reshape = terrain_data['vehicle_cost'].values.reshape(self.grid_resolution, self.grid_resolution)
                
                # Find obstacle areas (high cost)
                obstacle_mask = cost_reshape > 0.8
                
                # Label connected components
                labeled_obstacles, num_obstacles = label(obstacle_mask)
                
                # Extract obstacle properties
                obstacles = []
                
                for obstacle_id in range(1, num_obstacles + 1):
                    # Get obstacle mask
                    obstacle_area = labeled_obstacles == obstacle_id
                    
                    # Calculate size
                    size = np.sum(obstacle_area)
                    
                    # Skip very small obstacles
                    if size < 10:
                        continue
                    
                    # Calculate centroid
                    iy, ix = np.where(obstacle_area)
                    centroid_y = np.mean(iy)
                    centroid_x = np.mean(ix)
                    
                    # Convert to lat/lon
                    lat_vals = np.linspace(self.lat_min, self.lat_max, self.grid_resolution)
                    lon_vals = np.linspace(self.lon_min, self.lon_max, self.grid_resolution)
                    
                    centroid_lat = lat_vals[int(round(centroid_y))]
                    centroid_lon = lon_vals[int(round(centroid_x))]
                    
                    # Add to obstacles
                    obstacles.append({
                        'obstacle_id': int(obstacle_id),
                        'coords': (float(centroid_lat), float(centroid_lon)),
                        'size': int(size)
                    })
                
                # Sort by size
                obstacles.sort(key=lambda x: x['size'], reverse=True)
                
                # Add to result
                result['obstacles'] = obstacles
            except:
                pass
        
        return result
    
    def generate_terrain_report(self, area_coords=None, target_type='infantry'):
        """
        Generate a comprehensive terrain report for military planning.
        
        Args:
            area_coords (list): List of (latitude, longitude) coordinates defining the area
            target_type (str): Target type for mobility assessment
            
        Returns:
            dict: Comprehensive terrain report
        """
        if self.terrain_grid is None:
            if self.verbose:
                print("Error: Terrain grid not created yet")
                print("Call create_terrain_grid() first")
            return None
        
        # Perform comprehensive terrain analysis
        # 1. Terrain classification
        terrain_class = self.classify_terrain(area_coords)
        
        if terrain_class is None:
            return None
        
        # 2. Key terrain analysis
        if area_coords:
            # Calculate area centroid
            area_lat = [p[0] for p in area_coords]
            area_lon = [p[1] for p in area_coords]
            
            centroid_lat = np.mean(area_lat)
            centroid_lon = np.mean(area_lon)
            
            # Assume enemy approach from the north (adjust as needed)
            enemy_direction = 0
            
            key_terrain = self.identify_key_terrain(enemy_approach_direction=enemy_direction)
            
            # Filter to the area
            if key_terrain is not None:
                lat_min, lat_max = min(area_lat), max(area_lat)
                lon_min, lon_max = min(area_lon), max(area_lon)
                
                mask = (
                    (key_terrain['latitude'] >= lat_min) &
                    (key_terrain['latitude'] <= lat_max) &
                    (key_terrain['longitude'] >= lon_min) &
                    (key_terrain['longitude'] <= lon_max)
                )
                
                key_terrain = key_terrain[mask]
        else:
            key_terrain = self.identify_key_terrain()
        
        # 3. Mobility corridors
        mobility_corridors = self.identify_mobility_corridors(target_type)
        
        # Filter corridors to the area if specified
        if area_coords and mobility_corridors:
            # Filter mobility corridors to those that intersect with the area
            filtered_corridors = []
            
            for corridor in mobility_corridors['corridors']:
                # Check if corridor centroid is within the area
                centroid = (corridor['centroid_lat'], corridor['centroid_lon'])
                
                # Create a polygon for the area
                from shapely.geometry import Point, Polygon # type: ignore
                
                try:
                    area_poly = Polygon(area_coords)
                    centroid_point = Point(centroid)
                    
                    if area_poly.contains(centroid_point):
                        filtered_corridors.append(corridor)
                except:
                    # Fall back to bounding box check
                    lat_min, lat_max = min(area_lat), max(area_lat)
                    lon_min, lon_max = min(area_lon), max(area_lon)
                    
                    if (lat_min <= centroid[0] <= lat_max and
                        lon_min <= centroid[1] <= lon_max):
                        filtered_corridors.append(corridor)
            
            mobility_corridors['corridors'] = filtered_corridors
            mobility_corridors['num_corridors'] = len(filtered_corridors)
        
        # 4. Optimal observation points
        if area_coords:
            # Find optimal observation points for the area
            observation_points = self.find_optimal_observation_points(
                area_coords,
                num_points=3,
                max_range_km=5.0
            )
        else:
            observation_points = None
        
        # Compile the terrain report
        report = {
            'report_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_type': target_type,
            'area_coords': area_coords,
            'terrain_classification': terrain_class,
            'key_terrain': key_terrain['key_terrain_score'].quantile(0.95) if key_terrain is not None else None,
            'key_terrain_features': terrain_class['key_terrain_features'],
            'mobility_corridors': mobility_corridors['corridors'] if mobility_corridors else None,
            'optimal_observation_points': observation_points['observation_points'] if observation_points else None,
            'observation_coverage': observation_points['total_coverage_pct'] if observation_points else None
        }
        
        # Add OCOKA assessment (Observation, Cover and Concealment, Obstacles, Key Terrain, Avenues of Approach)
        report['ocoka_assessment'] = {
            'observation': self._assess_observation(terrain_class, key_terrain),
            'cover_concealment': self._assess_cover_concealment(terrain_class),
            'obstacles': self._assess_obstacles(terrain_class),
            'key_terrain': self._assess_key_terrain(terrain_class),
            'avenues_approach': self._assess_avenues_approach(mobility_corridors)
        }
        
        # Add KOCOA assessment (Key Terrain, Observation, Cover and Concealment, Obstacles, Avenues of Approach)
        # This is similar to OCOKA but with a different emphasis
        report['kocoa_assessment'] = report['ocoka_assessment'].copy()
        
        # Add METT-TC terrain assessment (Mission, Enemy, Terrain, Troops, Time, Civilian Considerations)
        report['mett_tc_terrain'] = self._assess_mett_tc(terrain_class, target_type)
        
        return report
    
    def _assess_observation(self, terrain_class, key_terrain):
        """Helper method to assess observation and fields of fire."""
        result = {
            'rating': '',
            'description': '',
            'favorable_areas': [],
            'unfavorable_areas': []
        }
        
        # Assess based on terrain classification
        if 'cover' in terrain_class['terrain_classification']:
            cover_level = terrain_class['terrain_classification']['cover']
            
            if cover_level == 'excellent':
                result['rating'] = 'restricted'
                result['description'] = 'Heavily restricted observation due to abundant vegetation and/or urban areas'
            elif cover_level == 'good':
                result['rating'] = 'limited'
                result['description'] = 'Limited observation with significant dead spaces'
            elif cover_level == 'moderate':
                result['rating'] = 'moderate'
                result['description'] = 'Moderate observation with some dead spaces'
            else:
                result['rating'] = 'excellent'
                result['description'] = 'Excellent observation and fields of fire'
        
        # Add relief influence
        if 'relief' in terrain_class['terrain_classification']:
            relief = terrain_class['terrain_classification']['relief']
            
            if relief == 'mountainous':
                if result['rating'] == 'excellent':
                    result['rating'] = 'variable'
                    result['description'] += ' from high ground, but many dead spaces in valleys'
                else:
                    result['description'] += ', further limited by mountainous terrain'
            elif relief == 'hilly':
                if result['rating'] == 'excellent':
                    result['rating'] = 'good'
                    result['description'] += ', with some limitations from hilly terrain'
            
        # Add information about favorable observation areas
        if key_terrain is not None and 'observation_score' in key_terrain.columns:
            # Find high observation score areas
            high_obs = key_terrain[key_terrain['observation_score'] > 0.7]
            
            if not high_obs.empty:
                # Get top 5 observation points
                top_obs = high_obs.sort_values('observation_score', ascending=False).head(5)
                
                for _, point in top_obs.iterrows():
                    result['favorable_areas'].append({
                        'coords': (point['latitude'], point['longitude']),
                        'elevation': point['elevation'],
                        'observation_score': point['observation_score']
                    })
        
        return result
    
    def _assess_cover_concealment(self, terrain_class):
        """Helper method to assess cover and concealment."""
        result = {
            'rating': '',
            'description': '',
            'favorable_areas': [],
            'unfavorable_areas': []
        }
        
        # Direct rating from terrain classification
        if 'cover' in terrain_class['terrain_classification']:
            cover_level = terrain_class['terrain_classification']['cover']
            
            result['rating'] = cover_level
            
            if cover_level == 'excellent':
                result['description'] = 'Abundant cover and concealment from vegetation and/or terrain features'
            elif cover_level == 'good':
                result['description'] = 'Good cover and concealment available throughout most of the area'
            elif cover_level == 'moderate':
                result['description'] = 'Moderate cover and concealment with some exposed areas'
            else:
                result['description'] = 'Poor cover and concealment with mostly exposed terrain'
        
        # Analyze land use distribution
        if 'land_use_distribution' in terrain_class:
            land_use = terrain_class['land_use_distribution']
            
            # Identify favorable areas
            favorable_types = ['forest', 'urban', 'light_vegetation']
            unfavorable_types = ['open_ground', 'road', 'water']
            
            for land_type in favorable_types:
                if land_type in land_use and land_use[land_type] > 15:
                    result['favorable_areas'].append({
                        'type': land_type,
                        'percentage': land_use[land_type]
                    })
            
            for land_type in unfavorable_types:
                if land_type in land_use and land_use[land_type] > 15:
                    result['unfavorable_areas'].append({
                        'type': land_type,
                        'percentage': land_use[land_type]
                    })
        
        return result
    
    def _assess_obstacles(self, terrain_class):
        """Helper method to assess obstacles."""
        result = {
            'rating': '',
            'description': '',
            'natural_obstacles': [],
            'restrictive_terrain': []
        }
        
        # Assess based on slope distribution
        if 'slope_stats' in terrain_class and 'distribution' in terrain_class['slope_stats']:
            slope_dist = terrain_class['slope_stats']['distribution']
            
            steep_pct = slope_dist.get('steep', 0) + slope_dist.get('very_steep', 0)
            
            if steep_pct > 40:
                result['rating'] = 'severely restricted'
                result['description'] = 'Movement severely restricted by steep slopes'
                
                result['natural_obstacles'].append({
                    'type': 'steep_slopes',
                    'percentage': steep_pct
                })
            elif steep_pct > 20:
                result['rating'] = 'restricted'
                result['description'] = 'Movement restricted by steep slopes in some areas'
                
                result['natural_obstacles'].append({
                    'type': 'steep_slopes',
                    'percentage': steep_pct
                })
        
        # Assess based on land use
        if 'land_use_distribution' in terrain_class:
            land_use = terrain_class['land_use_distribution']
            
            # Water obstacles
            if 'water' in land_use and land_use['water'] > 5:
                if result['rating']:
                    result['rating'] += ' and water'
                    result['description'] += f" and water bodies ({land_use['water']:.1f}% of area)"
                else:
                    result['rating'] = 'restricted'
                    result['description'] = f"Movement restricted by water bodies ({land_use['water']:.1f}% of area)"
                
                result['natural_obstacles'].append({
                    'type': 'water',
                    'percentage': land_use['water']
                })
            
            # Dense forest
            if 'forest' in land_use and land_use['forest'] > 30:
                if result['rating']:
                    result['rating'] += ' and vegetation'
                    result['description'] += f" and dense forest ({land_use['forest']:.1f}% of area)"
                else:
                    result['rating'] = 'restricted'
                    result['description'] = f"Movement restricted by dense forest ({land_use['forest']:.1f}% of area)"
                
                result['restrictive_terrain'].append({
                    'type': 'dense_forest',
                    'percentage': land_use['forest']
                })
            
            # Urban areas
            if 'urban' in land_use and land_use['urban'] > 20:
                if result['rating']:
                    result['rating'] += ' and urban'
                    result['description'] += f" and urban areas ({land_use['urban']:.1f}% of area)"
                else:
                    result['rating'] = 'channelized'
                    result['description'] = f"Movement channelized by urban areas ({land_use['urban']:.1f}% of area)"
                
                result['restrictive_terrain'].append({
                    'type': 'urban',
                    'percentage': land_use['urban']
                })
        
        # Add specific obstacles if available
        if 'obstacles' in terrain_class:
            for obstacle in terrain_class['obstacles'][:5]:  # Top 5 obstacles
                result['natural_obstacles'].append({
                    'type': 'terrain_obstacle',
                    'location': obstacle['coords'],
                    'size': obstacle['size']
                })
        
        # Set default rating if none assigned
        if not result['rating']:
            result['rating'] = 'unrestricted'
            result['description'] = 'No significant obstacles to movement'
        
        return result
    
    def _assess_key_terrain(self, terrain_class):
        """Helper method to assess key terrain."""
        result = {
            'rating': '',
            'description': '',
            'key_features': []
        }
        
        # Check if key terrain features were identified
        if 'key_terrain_features' in terrain_class and terrain_class['key_terrain_features']:
            features = terrain_class['key_terrain_features']
            
            # Count significant features
            significant_features = [f for f in features if f['score'] > 0.7]
            
            if len(significant_features) > 3:
                result['rating'] = 'abundant'
                result['description'] = f"Multiple significant key terrain features ({len(significant_features)} major features)"
            elif len(significant_features) > 0:
                result['rating'] = 'present'
                result['description'] = f"{len(significant_features)} key terrain features identified"
            else:
                result['rating'] = 'limited'
                result['description'] = "Limited key terrain in the area"
            
            # Add top key terrain features
            result['key_features'] = features[:5]  # Top 5 features
        else:
            result['rating'] = 'undetermined'
            result['description'] = "Key terrain analysis not performed"
        
        return result
    
    def _assess_avenues_approach(self, mobility_corridors):
        """Helper method to assess avenues of approach."""
        result = {
            'rating': '',
            'description': '',
            'corridors': []
        }
        
        # Check if mobility corridors were identified
        if mobility_corridors and 'corridors' in mobility_corridors and mobility_corridors['corridors']:
            corridors = mobility_corridors['corridors']
            
            # Count significant corridors
            large_corridors = [c for c in corridors if c['area'] > 100]
            
            if len(large_corridors) > 3:
                result['rating'] = 'multiple'
                result['description'] = f"Multiple avenues of approach available ({len(large_corridors)} major corridors)"
            elif len(large_corridors) > 0:
                result['rating'] = 'limited'
                result['description'] = f"{len(large_corridors)} primary avenues of approach"
            else:
                result['rating'] = 'restricted'
                result['description'] = "Few viable avenues of approach"
            
            # Add top mobility corridors
            result['corridors'] = corridors[:5]  # Top 5 corridors
        else:
            result['rating'] = 'undetermined'
            result['description'] = "Avenue of approach analysis not performed"
        
        return result
    
    def _assess_mett_tc(self, terrain_class, target_type):
        """Helper method to assess terrain for METT-TC."""
        # METT-TC: Mission, Enemy, Terrain, Troops, Time, Civilian Considerations
        # We focus on terrain aspects relevant to military planning
        
        result = {
            'mobility': {},
            'observation_fields_of_fire': {},
            'cover_concealment': {},
            'key_terrain': {},
            'obstacles': {},
            'weather_effects': {}
        }
        
        # Mobility assessment
        if 'mobility_assessment' in terrain_class and target_type in terrain_class['mobility_assessment']:
            mob = terrain_class['mobility_assessment'][target_type]
            
            result['mobility'] = {
                'classification': mob['classification'],
                'mean_cost': mob['mean_cost'],
                'assessment': f"Mobility for {target_type}: {mob['classification'].upper()}"
            }
            
            # Add more specific mobility considerations
            if 'terrain_classification' in terrain_class:
                terrain_class_data = terrain_class['terrain_classification']
                
                # Trafficability
                if 'trafficability' in terrain_class_data:
                    result['mobility']['trafficability'] = terrain_class_data['trafficability']
                
                # Effect of relief
                if 'relief' in terrain_class_data:
                    relief = terrain_class_data['relief']
                    
                    if relief == 'mountainous':
                        result['mobility']['terrain_effect'] = 'severely restrictive'
                    elif relief == 'hilly':
                        result['mobility']['terrain_effect'] = 'restrictive'
                    elif relief == 'gently rolling':
                        result['mobility']['terrain_effect'] = 'slightly restrictive'
                    else:
                        result['mobility']['terrain_effect'] = 'unrestricted'
        
        # Observation and Fields of Fire
        result['observation_fields_of_fire'] = self._assess_observation(terrain_class, None)
        
        # Cover and Concealment
        result['cover_concealment'] = self._assess_cover_concealment(terrain_class)
        
        # Key Terrain
        result['key_terrain'] = self._assess_key_terrain(terrain_class)
        
        # Obstacles
        result['obstacles'] = self._assess_obstacles(terrain_class)
        
        # Weather effects on terrain
        # This is a simple assessment based on terrain characteristics
        result['weather_effects'] = {
            'rain': self._assess_rain_effects(terrain_class),
            'snow': self._assess_snow_effects(terrain_class),
            'fog': self._assess_fog_effects(terrain_class),
            'temperature': self._assess_temperature_effects(terrain_class)
        }
        
        return result
    
    def _assess_rain_effects(self, terrain_class):
        """Helper method to assess rain effects on terrain."""
        result = {
            'mobility_impact': '',
            'visibility_impact': '',
            'description': ''
        }
        
        # Assess based on land use and slope
        if 'land_use_distribution' in terrain_class:
            land_use = terrain_class['land_use_distribution']
            
            # Check for soil types prone to mud
            mud_prone = land_use.get('open_ground', 0) + land_use.get('light_vegetation', 0)
            
            if mud_prone > 60:
                result['mobility_impact'] = 'severe'
                result['description'] = 'Rain likely to create significant mud and reduce mobility'
            elif mud_prone > 30:
                result['mobility_impact'] = 'moderate'
                result['description'] = 'Rain may create muddy conditions in parts of the area'
            else:
                result['mobility_impact'] = 'minimal'
                result['description'] = 'Limited mud-prone terrain'
        
        # Slope affects runoff and flood risk
        if 'slope_stats' in terrain_class:
            slope_mean = terrain_class['slope_stats'].get('mean_deg', 0)
            
            if slope_mean > 15:
                result['mobility_impact'] = 'moderate'  # Rapid runoff, potential for flash flooding
                
                if result['description']:
                    result['description'] += '; steep terrain creates rapid runoff and potential for flash flooding'
                else:
                    result['description'] = 'Steep terrain creates rapid runoff and potential for flash flooding'
            elif slope_mean < 5 and 'low' not in result['mobility_impact']:
                result['mobility_impact'] = 'moderate'  # Poor drainage, potential for ponding
                
                if result['description']:
                    result['description'] += '; flat terrain may have poor drainage and ponding'
                else:
                    result['description'] = 'Flat terrain may have poor drainage and ponding'
        
        # Visibility impact
        result['visibility_impact'] = 'moderate'  # Rain generally reduces visibility
        
        return result
    
    def _assess_snow_effects(self, terrain_class):
        """Helper method to assess snow effects on terrain."""
        result = {
            'mobility_impact': '',
            'visibility_impact': '',
            'description': ''
        }
        
        # Assess based on slope
        if 'slope_stats' in terrain_class:
            slope_mean = terrain_class['slope_stats'].get('mean_deg', 0)
            steep_pct = terrain_class['slope_stats'].get('distribution', {}).get('steep', 0) + \
                        terrain_class['slope_stats'].get('distribution', {}).get('very_steep', 0)
            
            if slope_mean > 15 or steep_pct > 30:
                result['mobility_impact'] = 'severe'
                result['description'] = 'Snow will make steep slopes hazardous and greatly reduce mobility'
            elif slope_mean > 8 or steep_pct > 15:
                result['mobility_impact'] = 'significant'
                result['description'] = 'Snow will create difficult conditions on slopes'
            else:
                result['mobility_impact'] = 'moderate'
                result['description'] = 'Snow will reduce mobility but terrain is mostly traversable'
        else:
            result['mobility_impact'] = 'moderate'
            result['description'] = 'Snow expected to reduce mobility'
        
        # Land use considerations
        if 'land_use_distribution' in terrain_class:
            land_use = terrain_class['land_use_distribution']
            
            if land_use.get('forest', 0) > 40:
                if result['description']:
                    result['description'] += '; forested areas will have less snow accumulation'
                else:
                    result['description'] = 'Forested areas will have less snow accumulation'
            
            if land_use.get('open_ground', 0) > 40:
                if result['description']:
                    result['description'] += '; open areas prone to snow drifts'
                else:
                    result['description'] = 'Open areas prone to snow drifts'
        
        # Visibility impact
        result['visibility_impact'] = 'varied'  # Snow can reduce visibility during storms but increase contrast after
        
        return result
    
    def _assess_fog_effects(self, terrain_class):
        """Helper method to assess fog effects on terrain."""
        result = {
            'likelihood': '',
            'visibility_impact': 'significant',
            'description': ''
        }
        
        # Assess based on elevation and land use
        if 'elevation_stats' in terrain_class and 'land_use_distribution' in terrain_class:
            elev_range = terrain_class['elevation_stats']['range_m']
            elev_mean = terrain_class['elevation_stats']['mean_m']
            land_use = terrain_class['land_use_distribution']
            
            # Valley fog conditions
            if elev_range > 200 and 'water' in land_use and land_use['water'] > 5:
                result['likelihood'] = 'high'
                result['description'] = 'Valleys and water bodies prone to fog formation'
            elif elev_range > 100:
                result['likelihood'] = 'moderate'
                result['description'] = 'Some valleys may experience fog'
            else:
                result['likelihood'] = 'variable'
                result['description'] = 'Fog possible but terrain not conducive to persistent fog'
        else:
            result['likelihood'] = 'undetermined'
            result['description'] = 'Insufficient data to assess fog conditions'
        
        return result
    
    def _assess_temperature_effects(self, terrain_class):
        """Helper method to assess temperature effects on terrain."""
        result = {
            'variations': '',
            'description': ''
        }
        
        # Assess based on elevation range and land use
        if 'elevation_stats' in terrain_class:
            elev_range = terrain_class['elevation_stats']['range_m']
            
            if elev_range > 500:
                result['variations'] = 'significant'
                result['description'] = f"Significant temperature variations due to {elev_range}m elevation range"
            elif elev_range > 200:
                result['variations'] = 'moderate'
                result['description'] = f"Moderate temperature variations due to {elev_range}m elevation range"
            else:
                result['variations'] = 'minimal'
                result['description'] = 'Minimal temperature variations due to elevation'
        else:
            result['variations'] = 'undetermined'
            result['description'] = 'Insufficient data to assess temperature variations'
        
        # Land use considerations
        if 'land_use_distribution' in terrain_class:
            land_use = terrain_class['land_use_distribution']
            
            if land_use.get('urban', 0) > 30:
                if result['description']:
                    result['description'] += '; urban heat island effect likely'
                else:
                    result['description'] = 'Urban heat island effect likely'
            
            if land_use.get('forest', 0) > 40:
                if result['description']:
                    result['description'] += '; forested areas more temperature-stable'
                else:
                    result['description'] = 'Forested areas more temperature-stable'
        
        return result


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    
    from data_loader import DataLoader
    from datetime import datetime
    
    # Load data
    loader = DataLoader(data_folder="./data", verbose=True)
    elevation_df = loader.load_elevation_map()
    land_use_df = loader.load_land_use()
    
    # Create terrain analyser
    terrain_analyser = TerrainAnalyser(
        elevation_grid=elevation_df,
        land_use_grid=land_use_df,
        verbose=True
    )
    
    # Create terrain grid
    terrain_grid = terrain_analyser.create_terrain_grid()
    
    # Create cost surface for vehicles
    vehicle_cost = terrain_analyser.create_cost_surface('vehicle')
    
    # Identify key terrain
    key_terrain = terrain_analyser.identify_key_terrain()
    
    # Find mobility corridors
    mobility_corridors = terrain_analyser.identify_mobility_corridors('vehicle')
    
    # Generate terrain report
    report = terrain_analyser.generate_terrain_report(target_type='infantry')
    
    print("Terrain analysis complete.")