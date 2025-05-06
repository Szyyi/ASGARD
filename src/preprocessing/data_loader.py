"""
Data Loader Module

This module handles the loading and initial preprocessing of all data sources required
for the predictive tracking system, including:
- Target sighting data (historical observations)
- Digital elevation models
- Land use/land cover data
- Blue force positions

It provides functionality to load from various file formats or generate synthetic data
for development and testing purposes.
"""

import os
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterio.transform import from_origin # type: ignore
from scipy.interpolate import griddata
from shapely.geometry import Point # type: ignore


class DataLoader:
    """
    A class responsible for loading and preprocessing data for the predictive tracking system.
    
    Attributes:
        data_folder (str): Path to the folder containing the data files
        random_seed (int): Seed for random number generation for reproducibility
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, data_folder="./data", random_seed=42, verbose=True):
        """
        Initialize the DataLoader.
        
        Args:
            data_folder (str): Path to the folder containing data files
            random_seed (int): Seed for random number generation
            verbose (bool): Whether to print detailed information
        """
        self.data_folder = data_folder
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)
        
        if self.verbose:
            print(f"DataLoader initialized with data folder: {data_folder}")
    
    def load_target_data(self, filename="target_data.csv"):
        """
        Load target sighting data from CSV file.
        
        The expected format includes columns for:
        - target_id: Unique identifier for each target
        - target_class: Type of target (e.g., vehicle, infantry)
        - latitude, longitude: Geographic coordinates
        - timestamp: Date and time of observation
        - Optional: speed, heading
        
        Args:
            filename (str): Name of the CSV file containing target data
            
        Returns:
            pd.DataFrame: DataFrame containing the target data
        """
        try:
            path = os.path.join(self.data_folder, filename)
            df = pd.read_csv(path)
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Handle potential variations in column names
            for expected, alternatives in [
                ('target_id', ['id', 'targetid', 'target']),
                ('target_class', ['class', 'type', 'targettype']),
                ('latitude', ['lat', 'y']),
                ('longitude', ['lon', 'long', 'x'])
            ]:
                # If the expected column is missing but an alternative exists, rename it
                if expected not in df.columns:
                    for alt in alternatives:
                        if alt in df.columns:
                            df.rename(columns={alt: expected}, inplace=True)
                            break
            
            # Validate that required columns exist
            required_cols = ['target_id', 'target_class', 'latitude', 'longitude', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                if self.verbose:
                    print(f"Warning: Missing required columns in {filename}: {missing_cols}")
                    print("Generating synthetic data instead.")
                return self._generate_sample_target_data()
            
            if self.verbose:
                print(f"Loaded {len(df)} target sightings from {filename}")
                print(f"Target classes: {df['target_class'].unique()}")
                print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except FileNotFoundError:
            if self.verbose:
                print(f"File {filename} not found, generating synthetic target data")
            return self._generate_sample_target_data()
        except Exception as e:
            if self.verbose:
                print(f"Error loading {filename}: {str(e)}")
                print("Generating synthetic data instead.")
            return self._generate_sample_target_data()
    
    def load_elevation_map(self, filename=None):
        """
        Load digital elevation model data.
        
        Can load from CSV, GeoTIFF, or other supported raster formats.
        
        Args:
            filename (str): Name of the file containing elevation data.
                           If None, will look for elevation.* in the data folder.
            
        Returns:
            pd.DataFrame: DataFrame with columns x, y, elevation
        """
        # If no filename is provided, look for elevation data files
        if filename is None:
            extensions = ['.csv', '.tif', '.tiff', '.asc', '.dem']
            for ext in extensions:
                potential_file = os.path.join(self.data_folder, f"elevation{ext}")
                if os.path.exists(potential_file):
                    filename = f"elevation{ext}"
                    break
        
        if filename is None:
            if self.verbose:
                print("No elevation file found, generating synthetic elevation data")
            return self._generate_sample_elevation()
        
        try:
            path = os.path.join(self.data_folder, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Load based on file extension
            if file_ext == '.csv':
                # Load from CSV
                df = pd.read_csv(path)
                
                # Handle potential variations in column names
                for expected, alternatives in [
                    ('x', ['lon', 'longitude', 'easting']),
                    ('y', ['lat', 'latitude', 'northing']),
                    ('elevation', ['elev', 'height', 'altitude', 'z'])
                ]:
                    if expected not in df.columns:
                        for alt in alternatives:
                            if alt in df.columns:
                                df.rename(columns={alt: expected}, inplace=True)
                                break
                
                # Check if we have the required columns
                if not all(col in df.columns for col in ['x', 'y', 'elevation']):
                    if self.verbose:
                        print(f"Missing required columns in {filename}")
                    return self._generate_sample_elevation()
                
            elif file_ext in ['.tif', '.tiff', '.asc', '.dem']:
                # Load from raster format using rasterio
                with rasterio.open(path) as src:
                    # Read the elevation data and mask
                    elevation = src.read(1)
                    mask = src.read_masks(1)
                    
                    # Get geographic coordinates for each pixel
                    height, width = elevation.shape
                    rows, cols = np.mgrid[0:height, 0:width]
                    
                    # Transform pixel coordinates to geographic coordinates
                    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                    
                    # Create a DataFrame with x, y, elevation
                    x_vals = np.array(xs).flatten()
                    y_vals = np.array(ys).flatten()
                    elev_vals = elevation.flatten()
                    mask_vals = mask.flatten()
                    
                    # Filter out masked (nodata) values
                    valid_mask = mask_vals != 0
                    df = pd.DataFrame({
                        'x': x_vals[valid_mask],
                        'y': y_vals[valid_mask],
                        'elevation': elev_vals[valid_mask]
                    })
            else:
                if self.verbose:
                    print(f"Unsupported file extension: {file_ext}")
                return self._generate_sample_elevation()
            
            if self.verbose:
                print(f"Loaded elevation data with {len(df)} points")
                print(f"Elevation range: {df['elevation'].min()} to {df['elevation'].max()}")
            
            return df
            
        except FileNotFoundError:
            if self.verbose:
                print(f"File {filename} not found, generating synthetic elevation data")
            return self._generate_sample_elevation()
        except Exception as e:
            if self.verbose:
                print(f"Error loading {filename}: {str(e)}")
                print("Generating synthetic elevation data instead.")
            return self._generate_sample_elevation()
    
    def load_land_use(self, filename=None):
        """
        Load land use / land cover data.
        
        Can load from CSV, GeoTIFF, or other supported raster formats.
        
        Args:
            filename (str): Name of the file containing land use data.
                           If None, will look for land_use.* in the data folder.
            
        Returns:
            pd.DataFrame: DataFrame with columns x, y, land_use, land_use_type
        """
        # If no filename is provided, look for land use data files
        if filename is None:
            extensions = ['.csv', '.tif', '.tiff', '.asc']
            for ext in extensions:
                potential_file = os.path.join(self.data_folder, f"land_use{ext}")
                if os.path.exists(potential_file):
                    filename = f"land_use{ext}"
                    break
        
        if filename is None:
            if self.verbose:
                print("No land use file found, generating synthetic land use data")
            return self._generate_sample_land_use()
        
        try:
            path = os.path.join(self.data_folder, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Define land use class mapping (adjust according to your data)
            land_use_map = {
                0: 'urban',
                1: 'road',
                2: 'forest',
                3: 'open',
                4: 'water',
                5: 'restricted',  # e.g., military zones
                6: 'wetland'
            }
            
            # Load based on file extension
            if file_ext == '.csv':
                # Load from CSV
                df = pd.read_csv(path)
                
                # Handle potential variations in column names
                for expected, alternatives in [
                    ('x', ['lon', 'longitude', 'easting']),
                    ('y', ['lat', 'latitude', 'northing']),
                    ('land_use', ['landuse', 'class', 'lulc', 'landcover'])
                ]:
                    if expected not in df.columns:
                        for alt in alternatives:
                            if alt in df.columns:
                                df.rename(columns={alt: expected}, inplace=True)
                                break
                
                # Check if we have the required columns
                if 'land_use' not in df.columns or 'x' not in df.columns or 'y' not in df.columns:
                    if self.verbose:
                        print(f"Missing required columns in {filename}")
                    return self._generate_sample_land_use()
                
                # Add text description of land use if not present
                if 'land_use_type' not in df.columns:
                    # Check if the land_use values match our expected codes
                    unique_codes = df['land_use'].unique()
                    missing_codes = [code for code in unique_codes if code not in land_use_map]
                    
                    if missing_codes:
                        if self.verbose:
                            print(f"Warning: Unknown land use codes: {missing_codes}")
                            print("These will be labeled as 'unknown'")
                        
                        # Add unknown codes to the mapping
                        for code in missing_codes:
                            land_use_map[code] = 'unknown'
                    
                    # Add the text descriptions
                    df['land_use_type'] = df['land_use'].map(land_use_map)
                
            elif file_ext in ['.tif', '.tiff', '.asc']:
                # Load from raster format using rasterio
                with rasterio.open(path) as src:
                    # Read the land use data and mask
                    land_use = src.read(1)
                    mask = src.read_masks(1)
                    
                    # Get geographic coordinates for each pixel
                    height, width = land_use.shape
                    rows, cols = np.mgrid[0:height, 0:width]
                    
                    # Transform pixel coordinates to geographic coordinates
                    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                    
                    # Create a DataFrame with x, y, land_use
                    x_vals = np.array(xs).flatten()
                    y_vals = np.array(ys).flatten()
                    lu_vals = land_use.flatten()
                    mask_vals = mask.flatten()
                    
                    # Filter out masked (nodata) values
                    valid_mask = mask_vals != 0
                    df = pd.DataFrame({
                        'x': x_vals[valid_mask],
                        'y': y_vals[valid_mask],
                        'land_use': lu_vals[valid_mask]
                    })
                    
                    # Add text description of land use
                    unique_codes = df['land_use'].unique()
                    missing_codes = [code for code in unique_codes if code not in land_use_map]
                    
                    if missing_codes:
                        if self.verbose:
                            print(f"Warning: Unknown land use codes: {missing_codes}")
                            print("These will be labeled as 'unknown'")
                        
                        # Add unknown codes to the mapping
                        for code in missing_codes:
                            land_use_map[code] = 'unknown'
                    
                    # Add the text descriptions
                    df['land_use_type'] = df['land_use'].map(land_use_map)
            else:
                if self.verbose:
                    print(f"Unsupported file extension: {file_ext}")
                return self._generate_sample_land_use()
            
            if self.verbose:
                print(f"Loaded land use data with {len(df)} points")
                print(f"Land use types: {df['land_use_type'].unique()}")
            
            return df
            
        except FileNotFoundError:
            if self.verbose:
                print(f"File {filename} not found, generating synthetic land use data")
            return self._generate_sample_land_use()
        except Exception as e:
            if self.verbose:
                print(f"Error loading {filename}: {str(e)}")
                print("Generating synthetic land use data instead.")
            return self._generate_sample_land_use()
    
    def export_to_geotiff(self, df, value_column, output_filename, resolution=100):
        """
        Export a DataFrame with x, y coordinates and values to a GeoTIFF file.
        
        Args:
            df (pd.DataFrame): DataFrame with x, y and value columns
            value_column (str): Name of the column containing the values to export
            output_filename (str): Name of the output GeoTIFF file
            resolution (int): Resolution of the output grid
            
        Returns:
            bool: Success indicator
        """
        try:
            # Get the bounds of the data
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            
            # Create a regular grid
            xi = np.linspace(x_min, x_max, resolution)
            yi = np.linspace(y_min, y_max, resolution)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate the values onto the grid
            points = df[['x', 'y']].values
            values = df[value_column].values
            zi_grid = griddata(points, values, (xi_grid, yi_grid), method='linear')
            
            # Calculate the pixel size
            pixel_size_x = (x_max - x_min) / (resolution - 1)
            pixel_size_y = (y_max - y_min) / (resolution - 1)
            
            # Create a GeoTIFF transform
            transform = from_origin(x_min, y_max, pixel_size_x, pixel_size_y)
            
            # Create the GeoTIFF
            output_path = os.path.join(self.data_folder, output_filename)
            new_dataset = rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=zi_grid.shape[0],
                width=zi_grid.shape[1],
                count=1,
                dtype=zi_grid.dtype,
                crs='+proj=latlong',
                transform=transform
            )
            
            # Write the data
            new_dataset.write(zi_grid, 1)
            new_dataset.close()
            
            if self.verbose:
                print(f"Exported {value_column} to {output_filename}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error exporting to GeoTIFF: {str(e)}")
            return False
    
    def _generate_sample_target_data(self, n_targets=10, n_records=100):
        """
        Generate synthetic target data for development and testing.
        
        Creates realistic movement patterns with terrain influence.
        
        Args:
            n_targets (int): Number of distinct targets to generate
            n_records (int): Total number of observation records
            
        Returns:
            pd.DataFrame: Synthetic target observation data
        """
        if self.verbose:
            print(f"Generating synthetic target data with {n_targets} targets and ~{n_records} records")
        
        # Define target classes with realistic characteristics
        target_classes = {
            'vehicle': {'speed_range': (10, 30), 'heading_std': 10, 'road_preference': 0.8},
            'infantry': {'speed_range': (2, 5), 'heading_std': 20, 'road_preference': 0.4},
            'artillery': {'speed_range': (5, 15), 'heading_std': 15, 'road_preference': 0.6},
            'command': {'speed_range': (8, 20), 'heading_std': 12, 'road_preference': 0.7}
        }
        
        # Define geographic bounds (using realistic coordinates)
        lat_min, lat_max = 34.0, 34.1
        lon_min, lon_max = -118.3, -118.2
        
        # Lists to store records
        records = []
        blue_force_records = []
        
        # Define start time (use a recent date)
        start_time = datetime.datetime(2025, 5, 1, 10, 0, 0)
        
        # Generate target data
        target_ids = list(range(1, n_targets+1))
        records_per_target = n_records // n_targets
        
        for target_id in target_ids:
            # Randomly select target class
            target_class = np.random.choice(list(target_classes.keys()))
            class_props = target_classes[target_class]
            
            # Set base speed based on target class
            base_speed = np.random.uniform(*class_props['speed_range'])
            
            # Random starting position
            current_lat = np.random.uniform(lat_min, lat_max)
            current_lon = np.random.uniform(lon_min, lon_max)
            
            # Initial heading (0-360 degrees, 0=North, 90=East)
            heading = np.random.uniform(0, 360)
            
            # Generate movement path with persistence in direction
            for i in range(records_per_target):
                # Add some randomness to heading
                heading_change = np.random.normal(0, class_props['heading_std'])
                heading = (heading + heading_change) % 360
                
                # Adjust speed randomly
                speed = np.random.normal(base_speed, base_speed * 0.1)
                speed = max(0, speed)  # Ensure non-negative speed
                
                # Calculate time delta (randomize observation intervals)
                time_delta = max(1, np.random.normal(5, 1))  # minutes
                
                # Calculate distance moved
                distance = speed * (time_delta / 60) / 111  # Convert km/h to degrees (approximate)
                
                # Calculate new position
                heading_rad = np.deg2rad(heading)
                lon_delta = distance * np.sin(heading_rad) / np.cos(np.deg2rad(current_lat))
                lat_delta = distance * np.cos(heading_rad)
                
                # Update position
                current_lat += lat_delta
                current_lon += lon_delta
                
                # Ensure position is within bounds
                current_lat = max(lat_min, min(lat_max, current_lat))
                current_lon = max(lon_min, min(lon_max, current_lon))
                
                # Calculate timestamp
                timestamp = start_time + datetime.timedelta(minutes=i*time_delta)
                
                # Create record
                records.append({
                    'target_id': target_id,
                    'target_class': target_class,
                    'latitude': current_lat,
                    'longitude': current_lon,
                    'timestamp': timestamp,
                    'speed': speed,
                    'heading': heading,
                    'is_blue': 0  # Flag to indicate this is a target (not a blue force)
                })
        
        # Generate blue force data
        blue_classes = ['infantry', 'recon', 'armor', 'command']
        n_blue_forces = max(1, n_targets // 2)  # Create fewer blue forces than targets
        
        for blue_id in range(1, n_blue_forces + 1):
            blue_class = np.random.choice(blue_classes)
            
            # Blue forces have more stable positions
            blue_lat = np.random.uniform(lat_min, lat_max)
            blue_lon = np.random.uniform(lon_min, lon_max)
            
            # Create records for each timestamp in the dataset
            timestamps = sorted(set(r['timestamp'] for r in records))
            
            for timestamp in timestamps:
                # Add small random movements
                blue_lat_current = blue_lat + np.random.normal(0, 0.001)
                blue_lon_current = blue_lon + np.random.normal(0, 0.001)
                
                # Ensure within bounds
                blue_lat_current = max(lat_min, min(lat_max, blue_lat_current))
                blue_lon_current = max(lon_min, min(lon_max, blue_lon_current))
                
                blue_force_records.append({
                    'blue_id': blue_id,
                    'target_id': blue_id,  # For compatibility with target schema
                    'blue_class': blue_class,
                    'target_class': blue_class,  # For compatibility with target schema
                    'latitude': blue_lat_current,
                    'longitude': blue_lon_current,
                    'timestamp': timestamp,
                    'is_blue': 1  # Flag to indicate this is a blue force
                })
        
        # Create DataFrame
        df_targets = pd.DataFrame(records)
        df_blue = pd.DataFrame(blue_force_records)
        
        # Combine into one DataFrame
        combined_df = pd.concat([df_targets, df_blue], ignore_index=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        
        # Create a GeoDataFrame for export to other formats if needed
        gdf = gpd.GeoDataFrame(
            combined_df,
            geometry=gpd.points_from_xy(combined_df.longitude, combined_df.latitude)
        )
        
        # Save to file
        output_path = os.path.join(self.data_folder, 'target_data.csv')
        combined_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"Generated synthetic target data with {len(combined_df)} records")
            print(f"Data saved to {output_path}")
        
        return combined_df
    
    def _generate_sample_elevation(self, size=100):
        """
        Generate synthetic elevation data.
        
        Creates a realistic terrain model with multiple hills, valleys, and ridges.
        
        Args:
            size (int): Resolution of the grid (size x size)
            
        Returns:
            pd.DataFrame: DataFrame with x, y, elevation columns
        """
        if self.verbose:
            print(f"Generating synthetic elevation data with {size}x{size} grid")
        
        # Create a grid of coordinates (normalized 0-1)
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate multiple hills and valleys using gaussian functions
        Z = np.zeros((size, size))
        
        # Add several hills
        for _ in range(4):
            cx, cy = np.random.uniform(0.1, 0.9, 2)
            height = np.random.uniform(200, 500)
            sigma = np.random.uniform(0.05, 0.15)
            Z += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # Add ridges (elongated hills)
        for _ in range(3):
            cx, cy = np.random.uniform(0.1, 0.9, 2)
            height = np.random.uniform(150, 300)
            sigma_x = np.random.uniform(0.05, 0.15)
            sigma_y = np.random.uniform(0.01, 0.05)
            angle = np.random.uniform(0, np.pi)
            
            # Rotate coordinates
            X_rot = (X - cx) * np.cos(angle) - (Y - cy) * np.sin(angle)
            Y_rot = (X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
            
            Z += height * np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        
        # Add valleys (negative gaussian)
        for _ in range(2):
            cx, cy = np.random.uniform(0.1, 0.9, 2)
            depth = np.random.uniform(100, 200)
            sigma = np.random.uniform(0.05, 0.1)
            Z -= depth * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # Add perlin noise for natural texture
        noise_scale = 8  # Higher values give more detailed noise
        noise = self._generate_perlin_noise(size, size, noise_scale)
        Z += 50 * noise  # Scale the noise
        
        # Add a base elevation to avoid negative values
        base_elevation = 300
        Z += base_elevation
        
        # Convert to DataFrame
        elevation_df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'elevation': Z.flatten()
        })
        
        # Save to file
        output_path = os.path.join(self.data_folder, 'elevation.csv')
        elevation_df.to_csv(output_path, index=False)
        
        # Export as GeoTIFF for visualization
        self.export_to_geotiff(elevation_df, 'elevation', 'elevation.tif')
        
        if self.verbose:
            print(f"Generated synthetic elevation data with {len(elevation_df)} points")
            print(f"Elevation range: {Z.min():.1f} to {Z.max():.1f} meters")
            print(f"Data saved to {output_path}")
        
        return elevation_df
    
    def _generate_sample_land_use(self, size=100):
        """
        Generate synthetic land use / land cover data.
        
        Creates a realistic pattern of urban areas, roads, forests, open terrain, and water bodies.
        
        Args:
            size (int): Resolution of the grid (size x size)
            
        Returns:
            pd.DataFrame: DataFrame with x, y, land_use, land_use_type columns
        """
        if self.verbose:
            print(f"Generating synthetic land use data with {size}x{size} grid")
        
        # Create a grid of coordinates (normalized 0-1)
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize land use grid (0: urban, 1: road, 2: forest, 3: open terrain, 4: water)
        Z = np.ones((size, size), dtype=int) * 3  # Default to open terrain
        
        # Create urban areas (clusters)
        for _ in range(3):
            center_x = np.random.uniform(0.1, 0.9)
            center_y = np.random.uniform(0.1, 0.9)
            radius = np.random.uniform(0.05, 0.1)
            
            # Create irregular urban shapes
            noise = np.random.normal(0, 0.02, X.shape)
            urban_mask = ((X - center_x)**2 + (Y - center_y)**2 + noise) < radius**2
            Z[urban_mask] = 0  # Urban
            
            # Add smaller urban clusters nearby (suburbs)
            n_suburbs = np.random.randint(2, 5)
            for _ in range(n_suburbs):
                suburb_x = center_x + np.random.normal(0, 0.05)
                suburb_y = center_y + np.random.normal(0, 0.05)
                suburb_radius = np.random.uniform(0.02, 0.04)
                
                noise = np.random.normal(0, 0.01, X.shape)
                suburb_mask = ((X - suburb_x)**2 + (Y - suburb_y)**2 + noise) < suburb_radius**2
                Z[suburb_mask] = 0  # Urban
        
        # Create roads (networks connecting urban areas and random paths)
        # Major roads (connect urban centers)
        urban_centers = []
        for i in range(size):
            for j in range(size):
                if Z[i, j] == 0:  # If urban
                    # Check if it's a center by looking at neighbors
                    is_center = True
                    for di in range(-3, 4):
                        for dj in range(-3, 4):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if Z[ni, nj] != 0:  # If not urban
                                    is_center = False
                                    break
                        if not is_center:
                            break
                    
                    if is_center:
                        urban_centers.append((i, j))
        
        # Connect urban centers with roads
        for idx1 in range(len(urban_centers)):
            for idx2 in range(idx1 + 1, len(urban_centers)):
                i1, j1 = urban_centers[idx1]
                i2, j2 = urban_centers[idx2]
                
                # Simplified A* path
                path = self._create_road_path(Z, i1, j1, i2, j2, size)
                
                # Apply the road
                for i, j in path:
                    # Create a road of width 1-2 pixels
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                # Don't overwrite urban areas
                                if Z[ni, nj] != 0:
                                    Z[ni, nj] = 1  # Road
        
        # Add some random roads
        for _ in range(5):
            i1, j1 = np.random.randint(0, size, 2)
            i2, j2 = np.random.randint(0, size, 2)
            
            path = self._create_road_path(Z, i1, j1, i2, j2, size)
            
            # Apply the road (narrower)
            for i, j in path:
                if 0 <= i < size and 0 <= j < size and Z[i, j] != 0:
                    Z[i, j] = 1  # Road
        
        # Create forests (irregular patches)
        for _ in range(8):
            center_x = np.random.uniform(0.1, 0.9)
            center_y = np.random.uniform(0.1, 0.9)
            radius = np.random.uniform(0.05, 0.15)
            
            # Create irregular shapes with perlin noise
            noise = self._generate_perlin_noise(size, size, 8) * 0.1
            forest_mask = ((X - center_x)**2 + (Y - center_y)**2 + noise) < radius**2
            
            # Only overwrite open terrain (not urban or roads)
            mask = forest_mask & (Z == 3)
            Z[mask] = 2  # Forest
        
        # Create water bodies (rivers and lakes)
        # First, create a few lakes
        for _ in range(2):
            center_x = np.random.uniform(0.1, 0.9)
            center_y = np.random.uniform(0.1, 0.9)
            radius = np.random.uniform(0.03, 0.08)
            
            # Create irregular lake shapes
            noise = self._generate_perlin_noise(size, size, 6) * 0.05
            lake_mask = ((X - center_x)**2 + (Y - center_y)**2 + noise) < radius**2
            
            # Only overwrite open terrain and forest (not urban or roads)
            mask = lake_mask & ((Z == 3) | (Z == 2))
            Z[mask] = 4  # Water
        
        # Then create rivers (curved paths)
        for _ in range(2):
            # Start at a random edge point
            edge = np.random.choice(4)  # 0: top, 1: right, 2: bottom, 3: left
            
            if edge == 0:
                start_i, start_j = 0, np.random.randint(0, size)
            elif edge == 1:
                start_i, start_j = np.random.randint(0, size), size - 1
            elif edge == 2:
                start_i, start_j = size - 1, np.random.randint(0, size)
            else:
                start_i, start_j = np.random.randint(0, size), 0
            
            # Find an end point (could be another edge or a lake)
            end_found = False
            while not end_found:
                edge_end = np.random.choice(4)
                
                if edge_end == 0:
                    end_i, end_j = 0, np.random.randint(0, size)
                elif edge_end == 1:
                    end_i, end_j = np.random.randint(0, size), size - 1
                elif edge_end == 2:
                    end_i, end_j = size - 1, np.random.randint(0, size)
                else:
                    end_i, end_j = np.random.randint(0, size), 0
                
                # Ensure start and end are different
                if (start_i != end_i or start_j != end_j) and edge != edge_end:
                    end_found = True
            
            # Create a river path
            river_points = self._create_river_path(start_i, start_j, end_i, end_j, size)
            
            # Apply the river with varying width
            for i, j in river_points:
                river_width = np.random.randint(1, 3)
                for di in range(-river_width, river_width + 1):
                    for dj in range(-river_width, river_width + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            # Rivers can run through anything except urban areas
                            if Z[ni, nj] != 0:
                                Z[ni, nj] = 4  # Water
        
        # Convert to DataFrame
        land_use_types = {0: 'urban', 1: 'road', 2: 'forest', 3: 'open', 4: 'water'}
        
        land_use_df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'land_use': Z.flatten()
        })
        
        # Add text description
        land_use_df['land_use_type'] = land_use_df['land_use'].map(land_use_types)
        
        # Save to file
        output_path = os.path.join(self.data_folder, 'land_use.csv')
        land_use_df.to_csv(output_path, index=False)
        
        # Export as GeoTIFF for visualization
        self.export_to_geotiff(land_use_df, 'land_use', 'land_use.tif')
        
        if self.verbose:
            print(f"Generated synthetic land use data with {len(land_use_df)} points")
            print(f"Land use categories: {land_use_types}")
            
            # Count occurrences of each land use type
            counts = land_use_df['land_use_type'].value_counts()
            for land_type, count in counts.items():
                percentage = 100 * count / len(land_use_df)
                print(f"  {land_type}: {count} points ({percentage:.1f}%)")
            
            print(f"Data saved to {output_path}")
        
        return land_use_df
    
    def _create_road_path(self, land_use, start_i, start_j, end_i, end_j, size):
        """
        Create a path for a road using a simplified A* approach.
        
        Args:
            land_use (numpy.ndarray): Land use grid
            start_i, start_j (int): Starting point indices
            end_i, end_j (int): Ending point indices
            size (int): Grid size
            
        Returns:
            list: List of (i, j) path points
        """
        # Simple implementation of A* algorithm with random variation
        path = []
        current_i, current_j = start_i, start_j
        path.append((current_i, current_j))
        
        # Maximum number of steps to prevent infinite loops
        max_steps = size * 2
        step_count = 0
        
        while (current_i != end_i or current_j != end_j) and step_count < max_steps:
            # Get direction to target
            di = np.sign(end_i - current_i)
            dj = np.sign(end_j - current_j)
            
            # Possible moves (including diagonals)
            moves = []
            for d_i in [di, 0, -di]:
                for d_j in [dj, 0, -dj]:
                    if d_i == 0 and d_j == 0:
                        continue
                    
                    ni, nj = current_i + d_i, current_j + d_j
                    if 0 <= ni < size and 0 <= nj < size:
                        # Prefer moving towards the target
                        priority = 0
                        if d_i == di:
                            priority += 2
                        if d_j == dj:
                            priority += 2
                        
                        # Prefer flat terrain
                        if land_use[ni, nj] == 3:  # Open terrain
                            priority += 1
                        elif land_use[ni, nj] == 0:  # Urban (don't go through centers)
                            priority -= 10
                        
                        # Add some randomness
                        priority += np.random.normal(0, 0.5)
                        
                        moves.append((ni, nj, priority))
            
            # Sort by priority (higher is better)
            moves.sort(key=lambda x: x[2], reverse=True)
            
            if moves:
                current_i, current_j = moves[0][0], moves[0][1]
                path.append((current_i, current_j))
            else:
                break  # No valid moves
            
            step_count += 1
        
        return path
    
    def _create_river_path(self, start_i, start_j, end_i, end_j, size):
        """
        Create a natural-looking river path with curves.
        
        Args:
            start_i, start_j (int): Starting point indices
            end_i, end_j (int): Ending point indices
            size (int): Grid size
            
        Returns:
            list: List of (i, j) path points
        """
        # Generate control points for a cubic Bezier curve
        control_points = [
            (start_i, start_j),
            (start_i + (end_i - start_i) // 3 + np.random.randint(-size//5, size//5),
             start_j + (end_j - start_j) // 3 + np.random.randint(-size//5, size//5)),
            (start_i + 2 * (end_i - start_i) // 3 + np.random.randint(-size//5, size//5),
             start_j + 2 * (end_j - start_j) // 3 + np.random.randint(-size//5, size//5)),
            (end_i, end_j)
        ]
        
        # Generate points along the curve
        num_points = max(size, int(np.sqrt((end_i - start_i)**2 + (end_j - start_j)**2) * 2))
        t = np.linspace(0, 1, num_points)
        
        path = []
        for ti in t:
            # Cubic Bezier formula
            point = (1-ti)**3 * np.array(control_points[0]) + \
                    3*(1-ti)**2*ti * np.array(control_points[1]) + \
                    3*(1-ti)*ti**2 * np.array(control_points[2]) + \
                    ti**3 * np.array(control_points[3])
            
            i, j = int(point[0]), int(point[1])
            if 0 <= i < size and 0 <= j < size:
                # Add some meanders
                i += int(np.sin(ti * 10) * size / 50)
                j += int(np.cos(ti * 10) * size / 50)
                
                # Ensure within bounds
                i = max(0, min(size-1, i))
                j = max(0, min(size-1, j))
                
                path.append((i, j))
        
        return path
    
    def _generate_perlin_noise(self, width, height, scale):
        """
        Generate Perlin noise for natural terrain features.
        
        This is a simplified version of Perlin noise suitable for our purpose.
        
        Args:
            width, height (int): Dimensions of the noise grid
            scale (float): Scale of the noise (higher values = more detailed)
            
        Returns:
            numpy.ndarray: Grid of noise values
        """
        def interpolate(a, b, t):
            # Cubic interpolation
            t = t * t * (3 - 2 * t)
            return a * (1 - t) + b * t
        
        # Generate a grid of random gradients
        angles = 2 * np.pi * np.random.rand(width + 1, height + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        
        # Initialize the noise grid
        noise = np.zeros((width, height))
        
        # Generate perlin noise
        for i in range(width):
            for j in range(height):
                # Get the coordinates in the gradient grid
                x, y = i / scale, j / scale
                
                # Get the integer coordinates
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + 1, y0 + 1
                
                # Get the fractional parts
                sx, sy = x - x0, y - y0
                
                # Get the dot products with the gradients
                n00 = np.dot(gradients[x0 % (width + 1), y0 % (height + 1)], [sx, sy])
                n10 = np.dot(gradients[x1 % (width + 1), y0 % (height + 1)], [sx - 1, sy])
                n01 = np.dot(gradients[x0 % (width + 1), y1 % (height + 1)], [sx, sy - 1])
                n11 = np.dot(gradients[x1 % (width + 1), y1 % (height + 1)], [sx - 1, sy - 1])
                
                # Interpolate
                nx0 = interpolate(n00, n10, sx)
                nx1 = interpolate(n01, n11, sx)
                noise[i, j] = interpolate(nx0, nx1, sy)
        
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        return noise


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(data_folder="./data", verbose=True)
    
    # Generate sample data
    targets_df = loader.load_target_data()
    elevation_df = loader.load_elevation_map()
    land_use_df = loader.load_land_use()
    
    print("Data generation complete.")