"""
Visualization Module

This module renders predictive tracking results in various visual formats,
including 2D heatmaps, 3D terrain visualizations, interactive maps, and
tactical overlays with military symbology.

Key capabilities:
- Probability heatmap visualization with confidence regions
- Interactive maps with multi-horizon predictions
- 3D terrain visualization with predicted paths
- Tactical overlays using military symbology
- Animation of predicted movement over time
- Comparison visualization of different prediction methods
"""

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import folium # type: ignore
from folium.plugins import HeatMap, MarkerCluster, MeasureControl, TimestampedGeoJson # type: ignore
from folium.features import DivIcon # type: ignore
from branca.colormap import LinearSegmentedColormap # type: ignore
import geopandas as gpd # type: ignore
from shapely.geometry import Point, LineString, MultiPoint, Polygon as ShapelyPolygon # type: ignore
from shapely.ops import cascaded_union # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
import networkx as nx # type: ignore
import io
import base64
from PIL import Image
import datetime
import os
import math
import json
import warnings
warnings.filterwarnings('ignore')


class Visualization:
    """
    Class for visualizing predictive tracking results.
    
    Attributes:
        targets_df (pd.DataFrame): DataFrame containing target observations
        blue_forces_df (pd.DataFrame): DataFrame containing blue force observations
        terrain_grid_df (pd.DataFrame): DataFrame containing terrain grid
        predictions (dict): Dictionary of prediction results
        config (dict): Configuration parameters
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, targets_df, blue_forces_df=None, terrain_grid_df=None, predictions=None, config=None, verbose=True):
        """
        Initialize the Visualization class.
        
        Args:
            targets_df (pd.DataFrame): DataFrame containing target observations
            blue_forces_df (pd.DataFrame): DataFrame containing blue force observations (optional)
            terrain_grid_df (pd.DataFrame): DataFrame containing terrain grid (optional)
            predictions (dict): Dictionary of prediction results (optional)
            config (dict): Configuration parameters (optional)
            verbose (bool): Whether to print detailed information
        """
        self.targets_df = targets_df
        self.blue_forces_df = blue_forces_df if blue_forces_df is not None else pd.DataFrame()
        self.terrain_grid_df = terrain_grid_df if terrain_grid_df is not None else pd.DataFrame()
        self.predictions = predictions if predictions is not None else {}
        self.verbose = verbose
        
        # Default configuration
        self.default_config = {
            'map_style': 'terrain',  # Base map style
            'heatmap_colors': ['blue', 'cyan', 'green', 'yellow', 'red'],  # Heatmap color gradient
            'show_confidence_regions': True,  # Whether to show confidence regions
            'confidence_levels': [68, 90, 95],  # Confidence region percentiles
            'terrain_cmap': 'terrain',  # Colormap for terrain elevation
            'landuse_colors': {  # Colors for different land use types
                'urban': '#A9A9A9',     # Dark gray
                'road': '#000000',      # Black
                'forest': '#228B22',    # Forest green
                'open': '#90EE90',      # Light green
                'water': '#1E90FF',     # Dodger blue
                'wetland': '#7D9EC0',   # Light slate
                'restricted': '#FF8C00' # Dark orange
            },
            'target_colors': {  # Colors for different target classes
                'vehicle': '#FF0000',    # Red
                'infantry': '#008000',   # Green
                'artillery': '#FFA500',  # Orange
                'command': '#0000FF'     # Blue
            },
            'blue_force_color': '#0000FF',  # Blue
            'prediction_opacity': 0.7,  # Opacity of prediction overlays
            'arrow_size': 10,  # Size of direction arrows
            'output_folder': './output',  # Folder for saving visualizations
            'dpi': 150,  # DPI for saved figures
            'fig_width': 12,  # Default figure width in inches
            'fig_height': 10,  # Default figure height in inches
            'animation_interval': 200,  # Animation interval in milliseconds
            'military_symbols': True,  # Whether to use military symbology
            'interactive': True,  # Whether to create interactive visualizations
            'north_arrow': True,  # Whether to show north arrow
            'scale_bar': True,  # Whether to show scale bar
            'grid_lines': False,  # Whether to show grid lines
            'title_fontsize': 16,  # Font size for titles
            'label_fontsize': 12,  # Font size for labels
            'legend_fontsize': 10,  # Font size for legends
            'timestamp_format': '%Y-%m-%d %H:%M',  # Format for timestamps
            'prediction_trail': True  # Whether to show prediction trail
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Create output folder if it doesn't exist
        if self.config['output_folder']:
            os.makedirs(self.config['output_folder'], exist_ok=True)
        
        # Set up custom colormaps
        self.setup_colormaps()
        
        if self.verbose:
            print("Visualization module initialized")
    
    def setup_colormaps(self):
        """
        Set up custom colormaps for visualizations.
        """
        # Heatmap colormap
        self.heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap', 
            self.config['heatmap_colors'], N=256)
        
        # Terrain colormap
        self.terrain_cmap = plt.cm.get_cmap(self.config['terrain_cmap'])
        
        # Land use colormap
        self.landuse_cmap = mcolors.ListedColormap(list(self.config['landuse_colors'].values()))
        
        # Confidence region colormap
        self.confidence_cmap = LinearSegmentedColormap.from_list('confidence', 
            [(0, (1, 0, 0, 0.3)), (0.5, (1, 0.5, 0, 0.3)), (1, (0, 0, 1, 0.3))], N=256)
    
    def plot_terrain(self, ax=None, show_elevation=True, show_landuse=True, alpha=0.5):
        """
        Plot terrain elevation and land use as background.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to plot on (optional)
            show_elevation (bool): Whether to show elevation
            show_landuse (bool): Whether to show land use
            alpha (float): Opacity of terrain layers
            
        Returns:
            matplotlib.axes.Axes: Axes with terrain plot
        """
        if self.terrain_grid_df.empty:
            if self.verbose:
                print("No terrain data available")
            return ax
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.config['fig_width'], self.config['fig_height']))
        
        # Check if we have required columns
        has_elevation = 'elevation' in self.terrain_grid_df.columns
        has_landuse = 'land_use_type' in self.terrain_grid_df.columns
        
        if not (has_elevation or has_landuse):
            if self.verbose:
                print("Missing elevation or land use columns")
            return ax
        
        # Get lat/lon bounds
        lat_min, lat_max = self.terrain_grid_df['latitude'].min(), self.terrain_grid_df['latitude'].max()
        lon_min, lon_max = self.terrain_grid_df['longitude'].min(), self.terrain_grid_df['longitude'].max()
        extent = [lon_min, lon_max, lat_min, lat_max]
        
        # Determine grid dimensions
        grid_size = int(np.sqrt(len(self.terrain_grid_df)))
        
        # Plot elevation if available and requested
        if has_elevation and show_elevation:
            # Reshape elevation to grid
            elevation = self.terrain_grid_df['elevation'].values.reshape(grid_size, grid_size)
            
            # Plot elevation as background
            im_elevation = ax.imshow(elevation, extent=extent, 
                                    cmap=self.terrain_cmap, alpha=alpha, origin='lower')
            
            # Add colorbar
            cbar = plt.colorbar(im_elevation, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label('Elevation (m)', fontsize=self.config['label_fontsize'])
        
        # Plot land use if available and requested
        if has_landuse and show_landuse:
            # Get unique land use types
            landuse_types = self.terrain_grid_df['land_use_type'].unique()
            
            # Create a mapping from type to integer
            landuse_map = {t: i for i, t in enumerate(landuse_types)}
            
            # Map land use types to integers
            self.terrain_grid_df['landuse_int'] = self.terrain_grid_df['land_use_type'].map(landuse_map)
            
            # Reshape land use to grid
            landuse = self.terrain_grid_df['landuse_int'].values.reshape(grid_size, grid_size)
            
            # Plot land use
            if show_elevation:
                # If elevation is shown, use alpha for land use
                im_landuse = ax.imshow(landuse, extent=extent, 
                                    cmap=self.landuse_cmap, alpha=0.5, origin='lower')
            else:
                # If elevation is not shown, use full opacity for land use
                im_landuse = ax.imshow(landuse, extent=extent, 
                                    cmap=self.landuse_cmap, alpha=1.0, origin='lower')
            
            # Add legend for land use
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=self.config['landuse_colors'].get(lt, '#FFFFFF'), 
                    edgecolor='black', label=lt)
                for lt in landuse_types
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     fontsize=self.config['legend_fontsize'], framealpha=0.7)
        
        # Add grid lines if requested
        if self.config['grid_lines']:
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add north arrow if requested
        if self.config['north_arrow']:
            self._add_north_arrow(ax)
        
        # Add scale bar if requested
        if self.config['scale_bar']:
            self._add_scale_bar(ax, lon_min, lat_min)
        
        return ax
    
    def _add_north_arrow(self, ax):
        """
        Add a north arrow to the plot.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to add arrow to
        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate arrow position (top right corner)
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.1
        arrow_length = (ylim[1] - ylim[0]) * 0.05
        
        # Draw arrow
        ax.arrow(x_pos, y_pos, 0, arrow_length, head_width=arrow_length/2, 
                head_length=arrow_length/2, fc='black', ec='black')
        
        # Add 'N' label
        ax.text(x_pos, y_pos + arrow_length * 1.2, 'N', 
               ha='center', va='bottom', fontsize=self.config['label_fontsize']*0.8)
    
    def _add_scale_bar(self, ax, lon_min, lat_min):
        """
        Add a scale bar to the plot.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to add scale bar to
            lon_min (float): Minimum longitude
            lat_min (float): Minimum latitude
        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate position (bottom left corner)
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05
        
        # Calculate scale bar length (aim for ~2 km)
        # Approximate conversion from degrees to km at this latitude
        lon_to_km = 111.32 * np.cos(np.radians(lat_min))
        
        # Find a round number of km for the scale bar
        km_options = [0.5, 1, 2, 5, 10]
        scale_km = min(km_options, key=lambda x: abs(x - 2))
        
        # Convert km to degrees longitude
        scale_degrees = scale_km / lon_to_km
        
        # Draw scale bar
        ax.plot([x_pos, x_pos + scale_degrees], [y_pos, y_pos], 'k-', linewidth=2)
        
        # Add ticks at ends
        tick_height = (ylim[1] - ylim[0]) * 0.01
        ax.plot([x_pos, x_pos], [y_pos - tick_height, y_pos + tick_height], 'k-', linewidth=1)
        ax.plot([x_pos + scale_degrees, x_pos + scale_degrees], 
               [y_pos - tick_height, y_pos + tick_height], 'k-', linewidth=1)
        
        # Add label
        ax.text(x_pos + scale_degrees / 2, y_pos + tick_height * 2, f'{scale_km} km', 
               ha='center', va='bottom', fontsize=self.config['label_fontsize']*0.8)
    
    def plot_target_trajectories(self, target_ids=None, ax=None, show_terrain=True, show_markers=True, 
                                show_labels=True, show_timestamps=False, show_blue_forces=True):
        """
        Plot historical trajectories of targets.
        
        Args:
            target_ids (list): List of target IDs to plot (optional, plots all if None)
            ax (matplotlib.axes.Axes): Axes to plot on (optional)
            show_terrain (bool): Whether to show terrain
            show_markers (bool): Whether to show markers
            show_labels (bool): Whether to show target labels
            show_timestamps (bool): Whether to show timestamps
            show_blue_forces (bool): Whether to show blue forces
            
        Returns:
            matplotlib.axes.Axes: Axes with trajectory plot
        """
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.config['fig_width'], self.config['fig_height']))
        
        # Plot terrain if requested
        if show_terrain and not self.terrain_grid_df.empty:
            ax = self.plot_terrain(ax=ax, alpha=0.5)
        
        # Filter targets if target_ids provided
        if target_ids is not None:
            targets_subset = self.targets_df[self.targets_df['target_id'].isin(target_ids)].copy()
        else:
            targets_subset = self.targets_df.copy()
        
        # Filter out blue forces from targets
        if 'is_blue' in targets_subset.columns:
            targets_subset = targets_subset[targets_subset['is_blue'] == 0].copy()
        
        # Get unique target IDs
        unique_targets = targets_subset['target_id'].unique()
        
        # Plot each target's trajectory
        for target_id in unique_targets:
            target_data = targets_subset[targets_subset['target_id'] == target_id].copy()
            
            # Skip if no data
            if target_data.empty:
                continue
            
            # Sort by timestamp
            target_data = target_data.sort_values('timestamp')
            
            # Get target class for color
            if 'target_class' in target_data.columns:
                target_class = target_data['target_class'].iloc[0]
                color = self.config['target_colors'].get(target_class, 'red')
            else:
                color = 'red'
                target_class = 'unknown'
            
            # Plot trajectory
            ax.plot(target_data['longitude'], target_data['latitude'], 
                   color=color, linewidth=2, alpha=0.8, 
                   label=f"Target {target_id} ({target_class})")
            
            # Plot markers if requested
            if show_markers:
                # Determine marker based on target class
                if target_class == 'vehicle':
                    marker = 's'  # square
                elif target_class == 'infantry':
                    marker = 'o'  # circle
                elif target_class == 'artillery':
                    marker = '^'  # triangle up
                else:
                    marker = 'D'  # diamond
                
                # Plot markers for each point
                ax.scatter(target_data['longitude'], target_data['latitude'], 
                          marker=marker, s=30, color=color, edgecolors='black', alpha=0.8)
            
            # Plot direction arrows if heading is available
            if 'heading' in target_data.columns and 'speed' in target_data.columns:
                # Plot arrows for selected points (not all to avoid cluttering)
                n_points = len(target_data)
                step = max(1, n_points // 5)  # Show at most 5 arrows
                
                for idx, row in target_data.iloc[::step].iterrows():
                    # Calculate arrow components
                    heading_rad = np.radians(row['heading'])
                    arrow_length = 0.0001 * row['speed']  # Scale by speed
                    dx = arrow_length * np.sin(heading_rad)
                    dy = arrow_length * np.cos(heading_rad)
                    
                    # Plot arrow
                    ax.arrow(row['longitude'], row['latitude'], dx, dy, 
                            head_width=arrow_length/3, head_length=arrow_length/2, 
                            fc=color, ec=color, alpha=0.8)
            
            # Plot labels if requested
            if show_labels:
                # Add label at start and end points
                if len(target_data) > 0:
                    # Start point
                    start_point = target_data.iloc[0]
                    ax.text(start_point['longitude'], start_point['latitude'], 
                           f"  Target {target_id} start", 
                           fontsize=self.config['label_fontsize']*0.8, color='black',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                    
                    # End point
                    end_point = target_data.iloc[-1]
                    ax.text(end_point['longitude'], end_point['latitude'], 
                           f"  Target {target_id} end", 
                           fontsize=self.config['label_fontsize']*0.8, color='black',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Plot timestamps if requested
            if show_timestamps:
                # Add timestamp at regular intervals
                n_points = len(target_data)
                step = max(1, n_points // 3)  # Show at most 3 timestamps
                
                for idx, row in target_data.iloc[::step].iterrows():
                    timestamp_str = row['timestamp'].strftime(self.config['timestamp_format'])
                    ax.text(row['longitude'], row['latitude'], f" {timestamp_str}", 
                           fontsize=self.config['label_fontsize']*0.7, color='darkblue',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
        
        # Plot blue forces if requested
        if show_blue_forces and not self.blue_forces_df.empty:
            # Get most recent position for each blue force
            if 'timestamp' in self.blue_forces_df.columns:
                blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            else:
                blue_latest = self.blue_forces_df.copy()
            
            # Plot blue forces
            for idx, row in blue_latest.iterrows():
                # Plot marker
                ax.scatter(row['longitude'], row['latitude'], 
                          marker='*', s=150, color=self.config['blue_force_color'], 
                          edgecolor='white', alpha=1.0, label='_nolegend_')
                
                # Plot label
                if show_labels:
                    blue_class = row['blue_class'] if 'blue_class' in row else 'unknown'
                    ax.text(row['longitude'], row['latitude'], 
                           f"  Blue Force {row['blue_id']} ({blue_class})", 
                           fontsize=self.config['label_fontsize']*0.8, color='blue',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Customize plot
        ax.set_xlabel('Longitude', fontsize=self.config['label_fontsize'])
        ax.set_ylabel('Latitude', fontsize=self.config['label_fontsize'])
        ax.set_title('Target Trajectories', fontsize=self.config['title_fontsize'])
        
        # Add legend (only for targets, not individual points)
        if len(unique_targets) > 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=self.config['legend_fontsize'])
        
        return ax
    
    def plot_prediction_heatmap(self, prediction_data, ax=None, show_terrain=True, 
                               show_trajectory=True, show_blue_forces=True, 
                               show_confidence_regions=None, alpha=0.7):
        """
        Plot prediction heatmap for a target.
        
        Args:
            prediction_data (dict): Prediction data from MovementPredictor
            ax (matplotlib.axes.Axes): Axes to plot on (optional)
            show_terrain (bool): Whether to show terrain
            show_trajectory (bool): Whether to show target trajectory
            show_blue_forces (bool): Whether to show blue forces
            show_confidence_regions (bool): Whether to show confidence regions (overrides config)
            alpha (float): Opacity of heatmap
            
        Returns:
            matplotlib.axes.Axes: Axes with heatmap plot
        """
        if prediction_data is None:
            if self.verbose:
                print("No prediction data provided")
            return None
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.config['fig_width'], self.config['fig_height']))
        
        # Plot terrain if requested
        if show_terrain and not self.terrain_grid_df.empty:
            ax = self.plot_terrain(ax=ax, alpha=0.3)
        
        # Extract prediction data
        lat_grid = prediction_data['lat_grid']
        lon_grid = prediction_data['lon_grid']
        density = prediction_data['density']
        minutes_ahead = prediction_data['minutes_ahead']
        target_id = prediction_data['target_id']
        
        # Get target class and last position
        target_class = prediction_data.get('target_class', 'unknown')
        last_lat = prediction_data.get('last_latitude', None)
        last_lon = prediction_data.get('last_longitude', None)
        last_heading = prediction_data.get('last_heading', None)
        last_speed = prediction_data.get('last_speed', None)
        
        # Create mesh for contour plot
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Plot heatmap
        contour = ax.contourf(lon_mesh, lat_mesh, density, levels=20, 
                            cmap=self.heatmap_cmap, alpha=alpha)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Probability Density', fontsize=self.config['label_fontsize'])
        
        # Plot confidence regions if requested
        if show_confidence_regions is None:
            show_confidence_regions = self.config['show_confidence_regions']
        
        if show_confidence_regions and 'confidence_regions' in prediction_data:
            conf_regions = prediction_data['confidence_regions']
            
            for level, region in conf_regions.items():
                # Get points for this confidence region
                if 'points' in region:
                    points = region['points']
                    
                    # Skip if not enough points to draw a polygon
                    if len(points) < 3:
                        continue
                    
                    # Create a convex hull around points
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(points)
                        
                        # Draw polygon
                        polygon = Polygon([points[i] for i in hull.vertices], 
                                        closed=True, alpha=0.3, 
                                        facecolor=self.confidence_cmap(level/100),
                                        edgecolor='black', linewidth=1, 
                                        label=f"{level}% Confidence")
                        ax.add_patch(polygon)
                    except:
                        # If ConvexHull fails, draw a scatter plot of points
                        lats = [p[0] for p in points]
                        lons = [p[1] for p in points]
                        ax.scatter(lons, lats, s=1, color=self.confidence_cmap(level/100), 
                                 alpha=0.3, label=f"{level}% Confidence")
        
        # Plot target trajectory if requested
        if show_trajectory and target_id is not None:
            # Get target data
            target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
            
            if not target_data.empty:
                # Sort by timestamp
                target_data = target_data.sort_values('timestamp')
                
                # Get color based on target class
                color = self.config['target_colors'].get(target_class, 'red')
                
                # Plot trajectory
                ax.plot(target_data['longitude'], target_data['latitude'], 
                       color=color, linewidth=2, alpha=0.8)
                
                # Plot last known position
                if last_lat is not None and last_lon is not None:
                    ax.scatter(last_lon, last_lat, 
                              marker='o', s=100, color=color, 
                              edgecolor='black', alpha=1.0, 
                              label=f"Last Known Position")
                    
                    # Add arrow showing heading if available
                    if last_heading is not None and last_speed is not None:
                        heading_rad = np.radians(last_heading)
                        arrow_length = 0.0001 * last_speed  # Scale by speed
                        dx = arrow_length * np.sin(heading_rad)
                        dy = arrow_length * np.cos(heading_rad)
                        
                        ax.arrow(last_lon, last_lat, dx, dy,
                                head_width=arrow_length/3, head_length=arrow_length/2, 
                                fc=color, ec=color, alpha=0.8)
        
        # Plot blue forces if requested
        if show_blue_forces and not self.blue_forces_df.empty:
            # Get most recent position for each blue force
            if 'timestamp' in self.blue_forces_df.columns:
                blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            else:
                blue_latest = self.blue_forces_df.copy()
            
            # Plot blue forces
            for idx, row in blue_latest.iterrows():
                # Plot marker
                ax.scatter(row['longitude'], row['latitude'], 
                          marker='*', s=150, color=self.config['blue_force_color'], 
                          edgecolor='white', alpha=1.0, label='_nolegend_')
                
                # Plot label
                blue_class = row['blue_class'] if 'blue_class' in row else 'unknown'
                ax.text(row['longitude'], row['latitude'], 
                       f"  Blue Force {row['blue_id']}", 
                       fontsize=self.config['label_fontsize']*0.8, color='blue',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Customize plot
        ax.set_xlabel('Longitude', fontsize=self.config['label_fontsize'])
        ax.set_ylabel('Latitude', fontsize=self.config['label_fontsize'])
        
        # Title
        title = f"Target {target_id} ({target_class}) - Predicted Location in {minutes_ahead} minutes"
        if 'method' in prediction_data:
            title += f" ({prediction_data['method']} method)"
        ax.set_title(title, fontsize=self.config['title_fontsize'])
        
        # Add legend
        if show_confidence_regions:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=self.config['legend_fontsize'])
        
        return ax
    
    def plot_multi_horizon_prediction(self, predictions, target_id, time_horizons=None, 
                                     method='integrated', grid_layout=None):
        """
        Plot predictions for multiple time horizons.
        
        Args:
            predictions (dict): Dictionary of prediction results
            target_id: Target ID to plot
            time_horizons (list): List of time horizons to plot (optional)
            method (str): Prediction method
            grid_layout (tuple): Grid layout as (rows, cols) (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure with multiple prediction plots
        """
        # Check if we have predictions
        if not predictions:
            if self.verbose:
                print("No predictions provided")
            return None
        
        # Filter predictions by target ID and method
        target_preds = {}
        for key, pred in predictions.items():
            if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                if 'method' in pred and pred['method'] == method:
                    horizon = pred['minutes_ahead']
                    target_preds[horizon] = pred
        
        if not target_preds:
            if self.verbose:
                print(f"No predictions found for target {target_id} using method {method}")
            return None
        
        # Use provided time horizons or all available
        if time_horizons is None:
            time_horizons = sorted(target_preds.keys())
        else:
            # Filter to available horizons
            time_horizons = [h for h in time_horizons if h in target_preds]
        
        if not time_horizons:
            if self.verbose:
                print(f"No predictions found for the specified time horizons")
            return None
        
        # Determine grid layout
        n_plots = len(time_horizons)
        if grid_layout is None:
            if n_plots <= 2:
                rows, cols = 1, n_plots
            elif n_plots <= 4:
                rows, cols = 2, 2
            elif n_plots <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = (n_plots + 2) // 3, 3  # Approximate a 3-column layout
        else:
            rows, cols = grid_layout
        
        # Create figure
        fig = plt.figure(figsize=(self.config['fig_width'] * cols / 2, 
                                 self.config['fig_height'] * rows / 2))
        
        # Get target information
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        if not target_data.empty:
            # Sort by timestamp
            target_data = target_data.sort_values('timestamp')
            
            # Get target class
            target_class = target_data['target_class'].iloc[0] if 'target_class' in target_data.columns else 'unknown'
        else:
            target_class = target_preds[time_horizons[0]].get('target_class', 'unknown')
        
        # Plot each time horizon
        for i, horizon in enumerate(time_horizons):
            if i < rows * cols:
                # Create subplot
                ax = fig.add_subplot(rows, cols, i+1)
                
                # Plot prediction heatmap
                pred = target_preds[horizon]
                self.plot_prediction_heatmap(pred, ax=ax, show_terrain=True, 
                                           show_trajectory=True, show_blue_forces=True, 
                                           show_confidence_regions=True, alpha=0.7)
                
                # Add horizon specific title
                ax.set_title(f"T+{horizon} min", fontsize=self.config['title_fontsize'])
        
        # Add overall title
        fig.suptitle(f"Multi-Horizon Prediction for Target {target_id} ({target_class})", 
                   fontsize=self.config['title_fontsize']+2)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        
        return fig
    
    def plot_prediction_comparison(self, predictions, target_id, time_horizon=30, methods=None):
        """
        Plot comparison of different prediction methods.
        
        Args:
            predictions (dict): Dictionary of prediction results
            target_id: Target ID to plot
            time_horizon (int): Time horizon to compare
            methods (list): List of methods to compare (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure with method comparison
        """
        # Check if we have predictions
        if not predictions:
            if self.verbose:
                print("No predictions provided")
            return None
        
        # Filter predictions by target ID and time horizon
        target_preds = {}
        for key, pred in predictions.items():
            if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                if 'minutes_ahead' in pred and pred['minutes_ahead'] == time_horizon:
                    if 'method' in pred:
                        method = pred['method']
                        target_preds[method] = pred
        
        if not target_preds:
            if self.verbose:
                print(f"No predictions found for target {target_id} at {time_horizon} minutes ahead")
            return None
        
        # Use provided methods or all available
        if methods is None:
            methods = sorted(target_preds.keys())
        else:
            # Filter to available methods
            methods = [m for m in methods if m in target_preds]
        
        if not methods:
            if self.verbose:
                print(f"No predictions found for the specified methods")
            return None
        
        # Determine grid layout
        n_plots = len(methods)
        if n_plots <= 2:
            rows, cols = 1, n_plots
        elif n_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        # Create figure
        fig = plt.figure(figsize=(self.config['fig_width'] * cols / 2, 
                                 self.config['fig_height'] * rows / 2))
        
        # Get target information
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        if not target_data.empty:
            # Sort by timestamp
            target_data = target_data.sort_values('timestamp')
            
            # Get target class
            target_class = target_data['target_class'].iloc[0] if 'target_class' in target_data.columns else 'unknown'
        else:
            target_class = target_preds[methods[0]].get('target_class', 'unknown')
        
        # Plot each method
        for i, method in enumerate(methods):
            if i < rows * cols:
                # Create subplot
                ax = fig.add_subplot(rows, cols, i+1)
                
                # Plot prediction heatmap
                pred = target_preds[method]
                self.plot_prediction_heatmap(pred, ax=ax, show_terrain=True, 
                                           show_trajectory=True, show_blue_forces=True, 
                                           show_confidence_regions=True, alpha=0.7)
                
                # Add method specific title
                ax.set_title(f"{method.capitalize()} Method", fontsize=self.config['title_fontsize'])
        
        # Add overall title
        fig.suptitle(f"Prediction Method Comparison for Target {target_id} ({target_class}) - T+{time_horizon} min", 
                   fontsize=self.config['title_fontsize']+2)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        
        return fig
    
    def create_interactive_map(self, prediction_data=None, target_ids=None, time_horizon=30, method='integrated',
                              width=800, height=600, center=None, zoom_start=12, marker_popups=True):
        """
        Create an interactive map with folium showing predictions.
        
        Args:
            prediction_data (dict): Prediction data (optional)
            target_ids (list): List of target IDs to include (optional)
            time_horizon (int): Time horizon for predictions (optional)
            method (str): Prediction method (optional)
            width (int): Map width in pixels
            height (int): Map height in pixels
            center (tuple): Map center coordinates (lat, lon) (optional)
            zoom_start (int): Initial zoom level
            marker_popups (bool): Whether to show popups on markers
            
        Returns:
            folium.Map: Interactive map
        """
        # Determine map center
        if center is None:
            # Use center of targets or blue forces
            if not self.targets_df.empty:
                center_lat = self.targets_df['latitude'].mean()
                center_lon = self.targets_df['longitude'].mean()
                center = (center_lat, center_lon)
            elif not self.blue_forces_df.empty:
                center_lat = self.blue_forces_df['latitude'].mean()
                center_lon = self.blue_forces_df['longitude'].mean()
                center = (center_lat, center_lon)
            else:
                # Default center
                center = (0, 0)
        
        # Create map
        m = folium.Map(location=center, 
                      zoom_start=zoom_start, 
                      tiles=self.config['map_style'],
                      width=width, 
                      height=height)
        
        # Add measure control for distance measurement
        m.add_child(MeasureControl())
        
        # Add tile layers
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Create feature groups for layers
        target_group = folium.FeatureGroup(name="Target Tracks")
        blue_group = folium.FeatureGroup(name="Blue Forces")
        prediction_group = folium.FeatureGroup(name=f"Predictions (T+{time_horizon} min)")
        
        # Add target trajectories
        if target_ids is None:
            # Use all targets
            if 'is_blue' in self.targets_df.columns:
                targets_subset = self.targets_df[self.targets_df['is_blue'] == 0].copy()
            else:
                targets_subset = self.targets_df.copy()
            
            target_ids = targets_subset['target_id'].unique()
        else:
            # Filter to specified targets
            targets_subset = self.targets_df[self.targets_df['target_id'].isin(target_ids)].copy()
            
            # Filter out blue forces
            if 'is_blue' in targets_subset.columns:
                targets_subset = targets_subset[targets_subset['is_blue'] == 0].copy()
        
        # Plot each target's trajectory
        for target_id in target_ids:
            target_data = targets_subset[targets_subset['target_id'] == target_id].copy()
            
            # Skip if no data
            if target_data.empty:
                continue
            
            # Sort by timestamp
            target_data = target_data.sort_values('timestamp')
            
            # Get target class for color
            if 'target_class' in target_data.columns:
                target_class = target_data['target_class'].iloc[0]
                color = self.config['target_colors'].get(target_class, 'red')
            else:
                color = 'red'
                target_class = 'unknown'
            
            # Create line for trajectory
            points = list(zip(target_data['latitude'], target_data['longitude']))
            if len(points) >= 2:
                folium.PolyLine(
                    points, 
                    color=color, 
                    weight=3, 
                    opacity=0.8, 
                    tooltip=f"Target {target_id} ({target_class})"
                ).add_to(target_group)
            
            # Add start marker
            if len(points) > 0:
                start_point = points[0]
                folium.Marker(
                    start_point, 
                    icon=folium.Icon(color='green', icon='play'),
                    tooltip=f"Target {target_id} start"
                ).add_to(target_group)
            
            # Add end marker
            if len(points) > 0:
                end_point = points[-1]
                
                # Create popup with target information
                if marker_popups:
                    last_row = target_data.iloc[-1]
                    popup_html = f"""
                    <div style="width: 200px">
                    <h4>Target {target_id}</h4>
                    <b>Class:</b> {target_class}<br>
                    <b>Last seen:</b> {last_row['timestamp'].strftime(self.config['timestamp_format'])}<br>
                    """
                    
                    if 'speed' in last_row:
                        popup_html += f"<b>Speed:</b> {last_row['speed']:.1f} km/h<br>"
                    
                    if 'heading' in last_row:
                        popup_html += f"<b>Heading:</b> {last_row['heading']:.1f}°<br>"
                    
                    if 'terrain_cost' in last_row:
                        popup_html += f"<b>Terrain:</b> {last_row['land_use_type'] if 'land_use_type' in last_row else 'unknown'}<br>"
                    
                    popup_html += "</div>"
                    
                    popup = folium.Popup(popup_html, max_width=300)
                else:
                    popup = None
                
                folium.Marker(
                    end_point, 
                    popup=popup,
                    icon=folium.Icon(color='red', icon='stop'),
                    tooltip=f"Target {target_id} last known position"
                ).add_to(target_group)
        
        # Add blue forces
        if not self.blue_forces_df.empty:
            # Get most recent position for each blue force
            if 'timestamp' in self.blue_forces_df.columns:
                blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            else:
                blue_latest = self.blue_forces_df.copy()
            
            # Create marker cluster for blue forces
            blue_cluster = MarkerCluster(name="Blue Forces").add_to(blue_group)
            
            # Add markers for each blue force
            for idx, row in blue_latest.iterrows():
                # Create popup with blue force information
                if marker_popups:
                    blue_class = row['blue_class'] if 'blue_class' in row else 'unknown'
                    popup_html = f"""
                    <div style="width: 200px">
                    <h4>Blue Force {row['blue_id']}</h4>
                    <b>Class:</b> {blue_class}<br>
                    """
                    
                    if 'timestamp' in row:
                        popup_html += f"<b>Last update:</b> {row['timestamp'].strftime(self.config['timestamp_format'])}<br>"
                    
                    popup_html += "</div>"
                    
                    popup = folium.Popup(popup_html, max_width=300)
                else:
                    popup = None
                
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=popup,
                    icon=folium.Icon(color='blue', icon='info-sign'),
                    tooltip=f"Blue Force {row['blue_id']}"
                ).add_to(blue_cluster)
        
        # Add predictions if available
        if prediction_data is not None:
            # If a dictionary of predictions is provided, extract the right ones
            if isinstance(prediction_data, dict) and 'density' not in prediction_data:
                # Check if we have predictions for the specified targets and time horizon
                relevant_preds = {}
                
                for key, pred in prediction_data.items():
                    if isinstance(pred, dict) and 'target_id' in pred:
                        if pred['target_id'] in target_ids:
                            if 'minutes_ahead' in pred and pred['minutes_ahead'] == time_horizon:
                                if 'method' in pred and pred['method'] == method:
                                    relevant_preds[pred['target_id']] = pred
                
                for target_id, pred in relevant_preds.items():
                    self._add_prediction_to_map(m, pred, prediction_group, marker_popups)
            else:
                # Single prediction provided
                self._add_prediction_to_map(m, prediction_data, prediction_group, marker_popups)
        
        # Add layers to map
        target_group.add_to(m)
        blue_group.add_to(m)
        prediction_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def _add_prediction_to_map(self, m, prediction_data, layer_group, marker_popups=True):
        """
        Add a prediction to the interactive map.
        
        Args:
            m (folium.Map): Map to add prediction to
            prediction_data (dict): Prediction data
            layer_group (folium.FeatureGroup): Layer group to add prediction to
            marker_popups (bool): Whether to show popups on markers
        """
        if prediction_data is None or 'density' not in prediction_data:
            return
        
        # Extract prediction data
        target_id = prediction_data['target_id']
        target_class = prediction_data.get('target_class', 'unknown')
        minutes_ahead = prediction_data['minutes_ahead']
        
        # Create heatmap from density grid
        lat_grid = prediction_data['lat_grid']
        lon_grid = prediction_data['lon_grid']
        density = prediction_data['density']
        
        # Convert to points for heatmap
        points = []
        intensities = []
        
        # Sample points from the density grid
        for i in range(len(lat_grid)):
            for j in range(len(lon_grid)):
                if density[i, j] > 0.01:  # Skip very low probability areas
                    points.append([lat_grid[i], lon_grid[j]])
                    intensities.append(float(density[i, j]))
        
        # Create sublayer for this prediction
        sublayer = folium.FeatureGroup(name=f"Target {target_id} - T+{minutes_ahead}min")
        
        # Add heatmap
        HeatMap(
            points,
            name=f"Target {target_id} Prediction",
            min_opacity=0.2,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.4: 'blue', 0.65: 'lime', 0.9: 'yellow', 1: 'red'},
            max_val=max(intensities) if intensities else 1.0,
            intensity=intensities
        ).add_to(sublayer)
        
        # Add confidence regions if available
        if 'confidence_regions' in prediction_data:
            conf_regions = prediction_data['confidence_regions']
            
            for level, region in conf_regions.items():
                # Get points for this confidence region
                if 'points' in region and len(region['points']) >= 3:
                    points = region['points']
                    
                    # Create a polygon for this confidence region
                    try:
                        # Try to create a convex hull
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(points)
                        hull_points = [points[i] for i in hull.vertices]
                        
                        # Add polygon
                        folium.Polygon(
                            locations=hull_points,
                            color='black',
                            weight=1,
                            fill=True,
                            fill_color=self._get_confidence_color(level),
                            fill_opacity=0.2,
                            tooltip=f"{level}% Confidence Region"
                        ).add_to(sublayer)
                    except:
                        # If convex hull fails, skip
                        pass
        
        # Add most likely position marker
        if 'density' in prediction_data:
            # Find maximum density point
            max_idx = np.argmax(prediction_data['density'])
            max_i, max_j = np.unravel_index(max_idx, prediction_data['density'].shape)
            max_lat = prediction_data['lat_grid'][max_i]
            max_lon = prediction_data['lon_grid'][max_j]
            
            # Create popup with prediction information
            if marker_popups:
                method = prediction_data.get('method', 'unknown')
                confidence = prediction_data.get('confidence', 0.0) * 100
                
                popup_html = f"""
                <div style="width: 200px">
                <h4>Target {target_id} Prediction</h4>
                <b>Time horizon:</b> {minutes_ahead} minutes<br>
                <b>Method:</b> {method}<br>
                <b>Confidence:</b> {confidence:.1f}%<br>
                <b>Class:</b> {target_class}<br>
                """
                
                # Add confidence region areas if available
                if 'confidence_regions' in prediction_data:
                    popup_html += "<b>Confidence regions:</b><br>"
                    for level, region in prediction_data['confidence_regions'].items():
                        if 'area_km2' in region:
                            popup_html += f"&nbsp;&nbsp;{level}%: {region['area_km2']:.2f} km²<br>"
                
                popup_html += "</div>"
                
                popup = folium.Popup(popup_html, max_width=300)
            else:
                popup = None
            
            # Add marker for most likely position
            color = self.config['target_colors'].get(target_class, 'red')
            folium.Marker(
                [max_lat, max_lon],
                popup=popup,
                icon=folium.Icon(color=color, icon='crosshairs'),
                tooltip=f"Target {target_id} - Most likely position at T+{minutes_ahead}min"
            ).add_to(sublayer)
        
        # Add sublayer to layer group
        sublayer.add_to(layer_group)
    
    def _get_confidence_color(self, level):
        """
        Get color for confidence region based on level.
        
        Args:
            level (int): Confidence level (e.g. 68, 90, 95)
            
        Returns:
            str: Hex color string
        """
        # Normalize level to 0-1 range
        norm_level = level / 100
        
        # Use confidence colormap
        rgb = self.confidence_cmap(norm_level)[:3]
        
        # Convert to hex
        hex_color = mcolors.rgb2hex(rgb)
        
        return hex_color
    
    def create_multi_time_interactive_map(self, predictions, target_ids=None, time_horizons=None, method='integrated',
                                         width=800, height=600, center=None, zoom_start=12):
        """
        Create an interactive map with predictions at multiple time horizons.
        
        Args:
            predictions (dict): Dictionary of prediction results
            target_ids (list): List of target IDs to include (optional)
            time_horizons (list): List of time horizons to include (optional)
            method (str): Prediction method (optional)
            width (int): Map width in pixels
            height (int): Map height in pixels
            center (tuple): Map center coordinates (lat, lon) (optional)
            zoom_start (int): Initial zoom level
            
        Returns:
            folium.Map: Interactive map
        """
        # Create base map
        m = self.create_interactive_map(None, target_ids, None, method, width, height, center, zoom_start)
        
        # Check if we have predictions
        if not predictions:
            return m
        
        # Get target IDs if not provided
        if target_ids is None:
            # Use all targets
            if 'is_blue' in self.targets_df.columns:
                targets_subset = self.targets_df[self.targets_df['is_blue'] == 0].copy()
            else:
                targets_subset = self.targets_df.copy()
            
            target_ids = targets_subset['target_id'].unique()
        
        # Use provided time horizons or all available
        if time_horizons is None:
            # Find all available time horizons
            available_horizons = set()
            for key, pred in predictions.items():
                if isinstance(pred, dict) and 'minutes_ahead' in pred:
                    available_horizons.add(pred['minutes_ahead'])
            
            time_horizons = sorted(available_horizons)
        
        # Create feature groups for each time horizon
        horizon_groups = {}
        for horizon in time_horizons:
            horizon_groups[horizon] = folium.FeatureGroup(name=f"T+{horizon} min Predictions")
        
        # Add predictions for each target and time horizon
        for target_id in target_ids:
            for horizon in time_horizons:
                # Find prediction for this target, horizon, and method
                for key, pred in predictions.items():
                    if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                        if 'minutes_ahead' in pred and pred['minutes_ahead'] == horizon:
                            if 'method' in pred and pred['method'] == method:
                                # Add prediction to map
                                self._add_prediction_to_map(m, pred, horizon_groups[horizon])
                                break
        
        # Add horizon groups to map (in reverse order so shorter horizons are on top)
        for horizon in sorted(time_horizons, reverse=True):
            horizon_groups[horizon].add_to(m)
        
        return m
    
    def create_3d_terrain_plot(self, prediction_data=None, target_id=None, time_horizon=30, method='integrated',
                              width=800, height=600, elevation_scale=0.0002, interactive=True, show_blue_forces=True):
        """
        Create a 3D terrain plot with prediction overlay.
        
        Args:
            prediction_data (dict): Prediction data (optional)
            target_id: Target ID to plot (optional)
            time_horizon (int): Time horizon for prediction (optional)
            method (str): Prediction method (optional)
            width (int): Plot width in pixels
            height (int): Plot height in pixels
            elevation_scale (float): Factor to scale elevation
            interactive (bool): Whether to create interactive plot
            show_blue_forces (bool): Whether to show blue forces
            
        Returns:
            plotly.graph_objects.Figure or matplotlib.figure.Figure: 3D terrain plot
        """
        # Check if we have terrain data
        if self.terrain_grid_df.empty or 'elevation' not in self.terrain_grid_df.columns:
            if self.verbose:
                print("No terrain data available for 3D plot")
            return None
        
        # Get prediction data if not provided but target_id is
        if prediction_data is None and target_id is not None:
            # Try to find prediction in self.predictions
            for key, pred in self.predictions.items():
                if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                    if 'minutes_ahead' in pred and pred['minutes_ahead'] == time_horizon:
                        if 'method' in pred and pred['method'] == method:
                            prediction_data = pred
                            break
        
        # Create grid for terrain
        grid_size = int(np.sqrt(len(self.terrain_grid_df)))
        
        # Reshape terrain data to grid
        terrain_lon = self.terrain_grid_df['longitude'].values.reshape(grid_size, grid_size)
        terrain_lat = self.terrain_grid_df['latitude'].values.reshape(grid_size, grid_size)
        terrain_elev = self.terrain_grid_df['elevation'].values.reshape(grid_size, grid_size)
        
        # Get land use data if available
        if 'land_use_type' in self.terrain_grid_df.columns:
            # Get unique land use types
            landuse_types = self.terrain_grid_df['land_use_type'].unique()
            
            # Create a mapping from type to integer
            landuse_map = {t: i for i, t in enumerate(landuse_types)}
            
            # Map land use types to integers
            self.terrain_grid_df['landuse_int'] = self.terrain_grid_df['land_use_type'].map(landuse_map)
            
            # Reshape land use to grid
            terrain_landuse = self.terrain_grid_df['landuse_int'].values.reshape(grid_size, grid_size)
            
            # Create a color mapping based on land use
            colormap = np.zeros((grid_size, grid_size, 4))  # RGBA
            
            for i, landuse_type in enumerate(landuse_types):
                if landuse_type in self.config['landuse_colors']:
                    color = mcolors.to_rgba(self.config['landuse_colors'][landuse_type])
                    mask = terrain_landuse == i
                    colormap[mask] = color
        else:
            # Use elevation for color
            colormap = None
        
        if interactive:
            # Create interactive 3D plot with Plotly
            fig = go.Figure()
            
            # Add terrain surface
            if colormap is not None:
                # Convert colormap to format acceptable by Plotly
                colorscale = []
                for i, landuse_type in enumerate(landuse_types):
                    if landuse_type in self.config['landuse_colors']:
                        color = self.config['landuse_colors'][landuse_type]
                        colorscale.append([i / len(landuse_types), color])
                
                fig.add_trace(go.Surface(
                    z=terrain_elev,
                    x=terrain_lon,
                    y=terrain_lat,
                    colorscale=colorscale,
                    showscale=False,
                    opacity=0.8
                ))
            else:
                fig.add_trace(go.Surface(
                    z=terrain_elev,
                    x=terrain_lon,
                    y=terrain_lat,
                    colorscale='terrain',
                    showscale=True,
                    opacity=0.8
                ))
            
            # Add prediction overlay if available
            if prediction_data is not None and 'density' in prediction_data:
                # Extract prediction data
                lat_grid = prediction_data['lat_grid']
                lon_grid = prediction_data['lon_grid']
                density = prediction_data['density']
                
                # Create mesh for prediction
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                
                # Interpolate elevation for prediction points
                from scipy.interpolate import griddata
                
                # Flatten terrain data for interpolation
                points = np.column_stack((terrain_lon.flatten(), terrain_lat.flatten()))
                values = terrain_elev.flatten()
                
                # Interpolate elevation
                pred_elev = griddata(points, values, (lon_mesh, lat_mesh), method='linear')
                
                # Add a small offset to display above terrain
                pred_elev = pred_elev + 10
                
                # Add prediction surface
                fig.add_trace(go.Surface(
                    z=pred_elev,
                    x=lon_mesh,
                    y=lat_mesh,
                    surfacecolor=density,
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(
                        title='Probability',
                        titleside='right'
                    )
                ))
            
            # Add target trajectory if available
            if target_id is not None:
                target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
                
                if not target_data.empty:
                    # Sort by timestamp
                    target_data = target_data.sort_values('timestamp')
                    
                    # Get target class for color
                    if 'target_class' in target_data.columns:
                        target_class = target_data['target_class'].iloc[0]
                        color = self.config['target_colors'].get(target_class, 'red')
                    else:
                        color = 'red'
                    
                    # Get trajectory points
                    traj_lon = target_data['longitude'].values
                    traj_lat = target_data['latitude'].values
                    
                    # Interpolate elevation for trajectory points
                    traj_elev = griddata(points, values, (traj_lon, traj_lat), method='linear')
                    
                    # Add a small offset to display above terrain
                    traj_elev = traj_elev + 20
                    
                    # Add trajectory line
                    fig.add_trace(go.Scatter3d(
                        x=traj_lon,
                        y=traj_lat,
                        z=traj_elev,
                        mode='lines+markers',
                        line=dict(
                            color=color,
                            width=5
                        ),
                        marker=dict(
                            size=5,
                            color=color
                        ),
                        name=f"Target {target_id} Trajectory"
                    ))
                    
                    # Add last known position marker
                    if len(target_data) > 0:
                        last_point = target_data.iloc[-1]
                        last_lon = last_point['longitude']
                        last_lat = last_point['latitude']
                        last_elev = traj_elev[-1] if len(traj_elev) > 0 else 0
                        
                        fig.add_trace(go.Scatter3d(
                            x=[last_lon],
                            y=[last_lat],
                            z=[last_elev],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=color,
                                symbol='diamond'
                            ),
                            name=f"Last Known Position"
                        ))
            
            # Add blue forces if available
            if show_blue_forces and not self.blue_forces_df.empty:
                # Get most recent position for each blue force
                if 'timestamp' in self.blue_forces_df.columns:
                    blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
                else:
                    blue_latest = self.blue_forces_df.copy()
                
                # Get blue force coordinates
                blue_lon = blue_latest['longitude'].values
                blue_lat = blue_latest['latitude'].values
                
                # Interpolate elevation for blue force points
                blue_elev = griddata(points, values, (blue_lon, blue_lat), method='linear')
                
                # Add a small offset to display above terrain
                blue_elev = blue_elev + 20
                
                # Add blue force markers
                fig.add_trace(go.Scatter3d(
                    x=blue_lon,
                    y=blue_lat,
                    z=blue_elev,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=self.config['blue_force_color'],
                        symbol='cross'
                    ),
                    name="Blue Forces"
                ))
            
            # Update layout
            fig.update_layout(
                title=f"3D Terrain with Predictions for Target {target_id}" if target_id else "3D Terrain Map",
                width=width,
                height=height,
                scene=dict(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    zaxis_title="Elevation (m)",
                    aspectratio=dict(x=1, y=1, z=elevation_scale)
                )
            )
            
            return fig
            
        else:
            # Create static 3D plot with Matplotlib
            fig = plt.figure(figsize=(self.config['fig_width'], self.config['fig_height']))
            ax = fig.add_subplot(111, projection='3d')
            
            # Add terrain surface
            if colormap is not None:
                # Plot the terrain with color based on land use
                surf = ax.plot_surface(
                    terrain_lon, terrain_lat, terrain_elev,
                    facecolors=colormap,
                    rstride=1, cstride=1,
                    alpha=0.8,
                    linewidth=0
                )
                
                # Add legend for land use
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=self.config['landuse_colors'].get(lt, '#FFFFFF'), 
                        edgecolor='black', label=lt)
                    for lt in landuse_types
                ]
                
                ax.legend(handles=legend_elements, loc='upper right', 
                         fontsize=self.config['legend_fontsize'])
            else:
                # Use elevation for color
                surf = ax.plot_surface(
                    terrain_lon, terrain_lat, terrain_elev,
                    cmap=self.terrain_cmap,
                    rstride=1, cstride=1,
                    alpha=0.8,
                    linewidth=0
                )
                
                # Add colorbar
                cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                cbar.set_label('Elevation (m)', fontsize=self.config['label_fontsize'])
            
            # Add prediction overlay if available
            if prediction_data is not None and 'density' in prediction_data:
                # Extract prediction data
                lat_grid = prediction_data['lat_grid']
                lon_grid = prediction_data['lon_grid']
                density = prediction_data['density']
                
                # Create mesh for prediction
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                
                # Interpolate elevation for prediction points
                from scipy.interpolate import griddata
                
                # Flatten terrain data for interpolation
                points = np.column_stack((terrain_lon.flatten(), terrain_lat.flatten()))
                values = terrain_elev.flatten()
                
                # Interpolate elevation
                pred_elev = griddata(points, values, (lon_mesh, lat_mesh), method='linear')
                
                # Add a small offset to display above terrain
                pred_elev = pred_elev + 10
                
                # Only show areas with significant probability
                mask = density > 0.1
                
                # Plot prediction as a colored surface
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(self.config['heatmap_colors'])
                
                prediction_surface = ax.plot_surface(
                    lon_mesh, lat_mesh, pred_elev,
                    facecolors=cmap(density),
                    rstride=1, cstride=1,
                    alpha=0.7,
                    linewidth=0,
                    zorder=10
                )
                
                # Add colorbar for prediction
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.1)
                cbar.set_label('Probability Density', fontsize=self.config['label_fontsize'])
            
            # Add target trajectory if available
            if target_id is not None:
                target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
                
                if not target_data.empty:
                    # Sort by timestamp
                    target_data = target_data.sort_values('timestamp')
                    
                    # Get target class for color
                    if 'target_class' in target_data.columns:
                        target_class = target_data['target_class'].iloc[0]
                        color = self.config['target_colors'].get(target_class, 'red')
                    else:
                        color = 'red'
                    
                    # Get trajectory points
                    traj_lon = target_data['longitude'].values
                    traj_lat = target_data['latitude'].values
                    
                    # Interpolate elevation for trajectory points
                    traj_elev = griddata(points, values, (traj_lon, traj_lat), method='linear')
                    
                    # Add a small offset to display above terrain
                    traj_elev = traj_elev + 20
                    
                    # Add trajectory line
                    ax.plot(
                        traj_lon, traj_lat, traj_elev,
                        color=color,
                        linewidth=3,
                        marker='o',
                        markersize=5,
                        label=f"Target {target_id} Trajectory"
                    )
                    
                    # Add last known position marker
                    if len(target_data) > 0:
                        last_point = target_data.iloc[-1]
                        last_lon = last_point['longitude']
                        last_lat = last_point['latitude']
                        last_elev = traj_elev[-1] if len(traj_elev) > 0 else 0
                        
                        ax.scatter(
                            [last_lon], [last_lat], [last_elev],
                            color=color,
                            s=100,
                            marker='D',
                            label=f"Last Known Position"
                        )
            
            # Add blue forces if available
            if not self.blue_forces_df.empty:
                # Get most recent position for each blue force
                if 'timestamp' in self.blue_forces_df.columns:
                    blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
                else:
                    blue_latest = self.blue_forces_df.copy()
                
                # Get blue force coordinates
                blue_lon = blue_latest['longitude'].values
                blue_lat = blue_latest['latitude'].values
                
                # Interpolate elevation for blue force points
                blue_elev = griddata(points, values, (blue_lon, blue_lat), method='linear')
                
                # Add a small offset to display above terrain
                blue_elev = blue_elev + 20
                
                # Add blue force markers
                ax.scatter(
                    blue_lon, blue_lat, blue_elev,
                    color=self.config['blue_force_color'],
                    s=100,
                    marker='*',
                    label="Blue Forces"
                )
            
            # Customize plot
            ax.set_xlabel('Longitude', fontsize=self.config['label_fontsize'])
            ax.set_ylabel('Latitude', fontsize=self.config['label_fontsize'])
            ax.set_zlabel('Elevation (m)', fontsize=self.config['label_fontsize'])
            
            # Title
            title = f"3D Terrain with Predictions for Target {target_id}" if target_id else "3D Terrain Map"
            ax.set_title(title, fontsize=self.config['title_fontsize'])
            
            # Add legend
            ax.legend(loc='upper right', fontsize=self.config['legend_fontsize'])
            
            # Set elevation scale
            ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, elevation_scale, 1]))
            
            return fig
    
    def create_animation(self, predictions, target_id, time_horizons=None, method='integrated',
                        fps=2, duration=10, show_terrain=True, show_blue_forces=True):
        """
        Create an animation of prediction evolution over time.
        
        Args:
            predictions (dict): Dictionary of prediction results
            target_id: Target ID to animate
            time_horizons (list): List of time horizons to include (optional)
            method (str): Prediction method (optional)
            fps (int): Frames per second
            duration (int): Animation duration in seconds
            show_terrain (bool): Whether to show terrain
            show_blue_forces (bool): Whether to show blue forces
            
        Returns:
            matplotlib.animation.FuncAnimation: Animation of predictions
        """
        # Filter predictions by target ID and method
        target_preds = {}
        for key, pred in predictions.items():
            if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                if 'method' in pred and pred['method'] == method:
                    horizon = pred['minutes_ahead']
                    target_preds[horizon] = pred
        
        if not target_preds:
            if self.verbose:
                print(f"No predictions found for target {target_id} using method {method}")
            return None
        
        # Use provided time horizons or all available
        if time_horizons is None:
            time_horizons = sorted(target_preds.keys())
        else:
            # Filter to available horizons
            time_horizons = [h for h in time_horizons if h in target_preds]
        
        if not time_horizons:
            if self.verbose:
                print(f"No predictions found for the specified time horizons")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.config['fig_width'], self.config['fig_height']))
        
        # Plot terrain if requested
        if show_terrain and not self.terrain_grid_df.empty:
            self.plot_terrain(ax=ax, alpha=0.3)
        
        # Get target trajectory
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        # Plot target trajectory
        if not target_data.empty:
            # Sort by timestamp
            target_data = target_data.sort_values('timestamp')
            
            # Get target class for color
            if 'target_class' in target_data.columns:
                target_class = target_data['target_class'].iloc[0]
                color = self.config['target_colors'].get(target_class, 'red')
            else:
                color = 'red'
                target_class = 'unknown'
            
            # Plot trajectory
            ax.plot(target_data['longitude'], target_data['latitude'], 
                   color=color, linewidth=2, alpha=0.8, label="Target Trajectory")
            
            # Plot markers for each point
            ax.scatter(target_data['longitude'], target_data['latitude'], 
                      marker='o', s=30, color=color, alpha=0.8)
            
            # Plot last known position
            if len(target_data) > 0:
                last_point = target_data.iloc[-1]
                ax.scatter(last_point['longitude'], last_point['latitude'], 
                          marker='D', s=100, color=color, alpha=1.0, 
                          label="Last Known Position")
        
        # Plot blue forces if requested
        if show_blue_forces and not self.blue_forces_df.empty:
            # Get most recent position for each blue force
            if 'timestamp' in self.blue_forces_df.columns:
                blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            else:
                blue_latest = self.blue_forces_df.copy()
            
            # Plot blue forces
            for idx, row in blue_latest.iterrows():
                # Plot marker
                ax.scatter(row['longitude'], row['latitude'], 
                          marker='*', s=150, color=self.config['blue_force_color'], 
                          edgecolor='white', alpha=1.0, label='_nolegend_')
        
        # Create a contour plot for the prediction
        # Initially empty
        contour = ax.contourf([[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], 
                            levels=20, cmap=self.heatmap_cmap, alpha=0.7)
        
        # Create text element for timestamp
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=self.config['label_fontsize'], 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Animation function
        def update(frame):
            # Clear previous contour
            for coll in contour.collections:
                coll.remove()
            
            # Get prediction for current frame
            horizon = time_horizons[frame % len(time_horizons)]
            pred = target_preds[horizon]
            
            # Extract prediction data
            lat_grid = pred['lat_grid']
            lon_grid = pred['lon_grid']
            density = pred['density']
            
            # Create mesh for contour plot
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Plot heatmap
            nonlocal contour
            contour = ax.contourf(lon_mesh, lat_mesh, density, levels=20, 
                                cmap=self.heatmap_cmap, alpha=0.7)
            
            # Update timestamp
            time_text.set_text(f"Prediction: T+{horizon} minutes")
            
            return [contour, time_text]
        
        # Create animation
        frames = len(time_horizons)
        interval = 1000 / fps  # milliseconds
        
        animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        
        # Customize plot
        ax.set_xlabel('Longitude', fontsize=self.config['label_fontsize'])
        ax.set_ylabel('Latitude', fontsize=self.config['label_fontsize'])
        
        # Title
        title = f"Prediction Evolution for Target {target_id} ({target_class})"
        ax.set_title(title, fontsize=self.config['title_fontsize'])
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=self.config['legend_fontsize'])
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Probability Density', fontsize=self.config['label_fontsize'])
        
        return animation
    
    def plot_military_overlay(self, predictions=None, target_ids=None, time_horizon=30, method='integrated',
                             show_terrain=True, show_blue_forces=True, show_symbols=True):
        """
        Plot predictions with military symbology overlay.
        
        Args:
            predictions (dict): Dictionary of prediction results (optional)
            target_ids (list): List of target IDs to include (optional)
            time_horizon (int): Time horizon for predictions (optional)
            method (str): Prediction method (optional)
            show_terrain (bool): Whether to show terrain
            show_blue_forces (bool): Whether to show blue forces
            show_symbols (bool): Whether to show military symbols
            
        Returns:
            matplotlib.figure.Figure: Figure with military overlay
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.config['fig_width'], self.config['fig_height']))
        
        # Plot terrain if requested
        if show_terrain and not self.terrain_grid_df.empty:
            self.plot_terrain(ax=ax, alpha=0.3)
        
        # Get target data
        if target_ids is None:
            # Use all targets
            if 'is_blue' in self.targets_df.columns:
                targets_subset = self.targets_df[self.targets_df['is_blue'] == 0].copy()
            else:
                targets_subset = self.targets_df.copy()
            
            target_ids = targets_subset['target_id'].unique()
        else:
            # Filter to specified targets
            targets_subset = self.targets_df[self.targets_df['target_id'].isin(target_ids)].copy()
            
            # Filter out blue forces
            if 'is_blue' in targets_subset.columns:
                targets_subset = targets_subset[targets_subset['is_blue'] == 0].copy()
        
        # Plot predictions if available
        if predictions is not None:
            for target_id in target_ids:
                # Find prediction for this target, horizon, and method
                for key, pred in predictions.items():
                    if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                        if 'minutes_ahead' in pred and pred['minutes_ahead'] == time_horizon:
                            if 'method' in pred and pred['method'] == method:
                                # Plot prediction
                                self.plot_prediction_heatmap(pred, ax=ax, show_terrain=False, 
                                                         show_trajectory=False, show_blue_forces=False, 
                                                         show_confidence_regions=True, alpha=0.7)
                                break
        
        # Plot each target with military symbology
        for target_id in target_ids:
            target_data = targets_subset[targets_subset['target_id'] == target_id].copy()
            
            # Skip if no data
            if target_data.empty:
                continue
            
            # Sort by timestamp
            target_data = target_data.sort_values('timestamp')
            
            # Get target class for symbol
            if 'target_class' in target_data.columns:
                target_class = target_data['target_class'].iloc[0]
            else:
                target_class = 'unknown'
            
            # Plot trajectory
            ax.plot(target_data['longitude'], target_data['latitude'], 
                   color='red', linewidth=2, alpha=0.8, linestyle='--')
            
            # Add military symbol at last position
            if len(target_data) > 0:
                last_point = target_data.iloc[-1]
                last_lon = last_point['longitude']
                last_lat = last_point['latitude']
                
                if show_symbols:
                    self._add_military_symbol(ax, last_lon, last_lat, target_class, 'hostile')
                else:
                    # Use simple marker if symbols not requested
                    marker = self._get_marker_for_class(target_class)
                    ax.scatter(last_lon, last_lat, marker=marker, s=100, color='red', 
                              edgecolor='black', alpha=1.0)
                
                # Add label
                ax.text(last_lon, last_lat + 0.002, f"Target {target_id}", 
                       ha='center', va='bottom', fontsize=self.config['label_fontsize']*0.8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Plot blue forces if requested
        if show_blue_forces and not self.blue_forces_df.empty:
            # Get most recent position for each blue force
            if 'timestamp' in self.blue_forces_df.columns:
                blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            else:
                blue_latest = self.blue_forces_df.copy()
            
            # Plot blue forces with military symbols
            for idx, row in blue_latest.iterrows():
                # Get blue force class
                blue_class = row['blue_class'] if 'blue_class' in row else 'unknown'
                
                # Add symbol
                if show_symbols:
                    self._add_military_symbol(ax, row['longitude'], row['latitude'], blue_class, 'friendly')
                else:
                    # Use simple marker if symbols not requested
                    ax.scatter(row['longitude'], row['latitude'], marker='*', s=150, 
                              color=self.config['blue_force_color'], edgecolor='black', alpha=1.0)
                
                # Add label
                ax.text(row['longitude'], row['latitude'] + 0.002, f"Blue {row['blue_id']}", 
                       ha='center', va='bottom', fontsize=self.config['label_fontsize']*0.8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add key terrain features if available
        self._add_key_terrain_features(ax)
        
        # Add compass rose
        if self.config['north_arrow']:
            self._add_north_arrow(ax)
        
        # Add scale bar
        if self.config['scale_bar']:
            lon_min, _ = ax.get_xlim()
            lat_min, _ = ax.get_ylim()
            self._add_scale_bar(ax, lon_min, lat_min)
        
        # Add timestamp
        current_time = datetime.datetime.now().strftime(self.config['timestamp_format'])
        ax.text(0.01, 0.01, f"Current time: {current_time}", transform=ax.transAxes,
               fontsize=self.config['label_fontsize']*0.8,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add prediction time
        ax.text(0.01, 0.04, f"Prediction: T+{time_horizon} minutes", transform=ax.transAxes,
               fontsize=self.config['label_fontsize']*0.8,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Customize plot
        ax.set_xlabel('Longitude', fontsize=self.config['label_fontsize'])
        ax.set_ylabel('Latitude', fontsize=self.config['label_fontsize'])
        
        # Title
        title = "Tactical Situation with Movement Predictions"
        ax.set_title(title, fontsize=self.config['title_fontsize'])
        
        return fig
    
    def _add_military_symbol(self, ax, lon, lat, unit_type, affiliation):
        """
        Add a military symbol to the plot using APP-6/MIL-STD-2525 style.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to add symbol to
            lon (float): Longitude
            lat (float): Latitude
            unit_type (str): Type of unit
            affiliation (str): Affiliation ('friendly', 'hostile', 'neutral', 'unknown')
        """
        # Set symbol properties based on affiliation
        if affiliation == 'friendly':
            frame_color = 'blue'
            fill_color = 'none'
            icon_color = 'blue'
        elif affiliation == 'hostile':
            frame_color = 'red'
            fill_color = 'none'
            icon_color = 'red'
        elif affiliation == 'neutral':
            frame_color = 'green'
            fill_color = 'none'
            icon_color = 'green'
        else:  # unknown
            frame_color = 'yellow'
            fill_color = 'none'
            icon_color = 'yellow'
        
        # Create frame based on affiliation
        if affiliation == 'friendly':
            # Friendly: Circle
            circle = plt.Circle((lon, lat), 0.0015, fill=False, 
                               edgecolor=frame_color, linewidth=2)
            ax.add_patch(circle)
        elif affiliation == 'hostile':
            # Hostile: Diamond
            diamond = plt.Polygon([
                (lon, lat + 0.002),
                (lon + 0.002, lat),
                (lon, lat - 0.002),
                (lon - 0.002, lat)
            ], closed=True, fill=False, edgecolor=frame_color, linewidth=2)
            ax.add_patch(diamond)
        elif affiliation == 'neutral':
            # Neutral: Square
            square = plt.Rectangle(
                (lon - 0.0015, lat - 0.0015),
                0.003, 0.003,
                fill=False, edgecolor=frame_color, linewidth=2
            )
            ax.add_patch(square)
        else:  # unknown
            # Unknown: Rectangle
            rectangle = plt.Rectangle(
                (lon - 0.002, lat - 0.0015),
                0.004, 0.003,
                fill=False, edgecolor=frame_color, linewidth=2
            )
            ax.add_patch(rectangle)
        
        # Add icon based on unit type
        if unit_type == 'infantry':
            # Infantry: Cross with horizontal line at bottom
            ax.plot([lon - 0.001, lon + 0.001], [lat, lat], color=icon_color, linewidth=2)
            ax.plot([lon, lon], [lat + 0.001, lat - 0.001], color=icon_color, linewidth=2)
        elif unit_type == 'vehicle' or unit_type == 'armor':
            # Vehicle: Oval
            ellipse = plt.Ellipse((lon, lat), 0.0015, 0.001, fill=False, 
                                 edgecolor=icon_color, linewidth=2)
            ax.add_patch(ellipse)
        elif unit_type == 'artillery':
            # Artillery: Circle with dot
            inner_circle = plt.Circle((lon, lat), 0.0007, fill=True, 
                                     color=icon_color, alpha=0.7)
            ax.add_patch(inner_circle)
        elif unit_type == 'command':
            # Command: Star
            ax.scatter(lon, lat, marker='*', s=60, color=icon_color, zorder=10)
        elif unit_type == 'recon':
            # Recon: Triangle
            triangle = plt.Polygon([
                (lon, lat + 0.001),
                (lon + 0.001, lat - 0.001),
                (lon - 0.001, lat - 0.001)
            ], closed=True, fill=False, edgecolor=icon_color, linewidth=2)
            ax.add_patch(triangle)
        else:
            # Unknown: X
            ax.plot([lon - 0.001, lon + 0.001], [lat - 0.001, lat + 0.001], 
                   color=icon_color, linewidth=2)
            ax.plot([lon - 0.001, lon + 0.001], [lat + 0.001, lat - 0.001], 
                   color=icon_color, linewidth=2)
    
    def _get_marker_for_class(self, unit_type):
        """
        Get marker symbol for unit type.
        
        Args:
            unit_type (str): Type of unit
            
        Returns:
            str: Matplotlib marker symbol
        """
        if unit_type == 'infantry':
            return 'o'  # Circle
        elif unit_type == 'vehicle':
            return 's'  # Square
        elif unit_type == 'artillery':
            return '^'  # Triangle up
        elif unit_type == 'command':
            return 'D'  # Diamond
        elif unit_type == 'recon':
            return 'v'  # Triangle down
        elif unit_type == 'armor':
            return 'h'  # Hexagon
        else:
            return 'X'  # X
    
    def _add_key_terrain_features(self, ax):
        """
        Add key terrain features to the plot.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to add features to
        """
        if self.terrain_grid_df.empty or 'land_use_type' not in self.terrain_grid_df.columns:
            return
        
        # Get key terrain types
        forests = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == 'forest']
        urban = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == 'urban']
        water = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == 'water']
        
        # Add urban areas
        if not urban.empty:
            # Group nearby urban points to create urban areas
            try:
                # Create points
                urban_points = [Point(row['longitude'], row['latitude']) for _, row in urban.iterrows()]
                
                # Buffer points to create areas
                urban_areas = [p.buffer(0.001) for p in urban_points]
                
                # Combine overlapping areas
                from shapely.ops import unary_union # type: ignore
                combined_urban = unary_union(urban_areas)
                
                # Add to plot
                if hasattr(combined_urban, 'geoms'):
                    # Multiple polygons
                    for poly in combined_urban.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color='gray', alpha=0.3, edgecolor='black', linewidth=1)
                else:
                    # Single polygon
                    x, y = combined_urban.exterior.xy
                    ax.fill(x, y, color='gray', alpha=0.3, edgecolor='black', linewidth=1)
            except:
                # Fallback: plot as points
                ax.scatter(urban['longitude'], urban['latitude'], color='gray', alpha=0.1, s=5)
        
        # Add forests
        if not forests.empty:
            # Group nearby forest points to create forest areas
            try:
                # Create points
                forest_points = [Point(row['longitude'], row['latitude']) for _, row in forests.iterrows()]
                
                # Buffer points to create areas
                forest_areas = [p.buffer(0.001) for p in forest_points]
                
                # Combine overlapping areas
                from shapely.ops import unary_union # type: ignore
                combined_forests = unary_union(forest_areas)
                
                # Add to plot
                if hasattr(combined_forests, 'geoms'):
                    # Multiple polygons
                    for poly in combined_forests.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color='darkgreen', alpha=0.2, edgecolor='darkgreen', linewidth=1)
                else:
                    # Single polygon
                    x, y = combined_forests.exterior.xy
                    ax.fill(x, y, color='darkgreen', alpha=0.2, edgecolor='darkgreen', linewidth=1)
            except:
                # Fallback: plot as points
                ax.scatter(forests['longitude'], forests['latitude'], color='darkgreen', alpha=0.1, s=5)
        
        # Add water bodies
        if not water.empty:
            # Group nearby water points to create water areas
            try:
                # Create points
                water_points = [Point(row['longitude'], row['latitude']) for _, row in water.iterrows()]
                
                # Buffer points to create areas
                water_areas = [p.buffer(0.001) for p in water_points]
                
                # Combine overlapping areas
                from shapely.ops import unary_union # type: ignore
                combined_water = unary_union(water_areas)
                
                # Add to plot
                if hasattr(combined_water, 'geoms'):
                    # Multiple polygons
                    for poly in combined_water.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color='blue', alpha=0.2, edgecolor='blue', linewidth=1)
                else:
                    # Single polygon
                    x, y = combined_water.exterior.xy
                    ax.fill(x, y, color='blue', alpha=0.2, edgecolor='blue', linewidth=1)
            except:
                # Fallback: plot as points
                ax.scatter(water['longitude'], water['latitude'], color='blue', alpha=0.1, s=5)
    
    def save_visualization(self, fig, filename=None, dpi=None, format='png'):
        """
        Save visualization to file.
        
        Args:
            fig: Figure to save
            filename (str): Output filename (optional)
            dpi (int): DPI for saved figure (optional)
            format (str): Output format (png, jpg, svg, pdf)
            
        Returns:
            str: Path to saved file
        """
        if dpi is None:
            dpi = self.config['dpi']
        
        if filename is None:
            # Generate filename based on current time
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_viz_{timestamp}.{format}"
        
        # Ensure output folder exists
        output_folder = self.config['output_folder']
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            filepath = os.path.join(output_folder, filename)
        else:
            filepath = filename
        
        # Save different types of figures
        if hasattr(fig, 'write_html'):
            # Plotly figure
            fig.write_html(filepath.replace(f'.{format}', '.html'))
            filepath = filepath.replace(f'.{format}', '.html')
        elif hasattr(fig, '_repr_html_'):
            # Folium map
            fig.save(filepath.replace(f'.{format}', '.html'))
            filepath = filepath.replace(f'.{format}', '.html')
        elif hasattr(fig, 'savefig'):
            # Matplotlib figure
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        else:
            if self.verbose:
                print(f"Unknown figure type, could not save")
            return None
        
        if self.verbose:
            print(f"Visualization saved to {filepath}")
        
        return filepath
    
    def generate_report(self, predictions, target_ids=None, time_horizons=None, methods=None, output_format='html'):
        """
        Generate a comprehensive report with multiple visualizations.
        
        Args:
            predictions (dict): Dictionary of prediction results
            target_ids (list): List of target IDs to include (optional)
            time_horizons (list): List of time horizons to include (optional)
            methods (list): List of methods to include (optional)
            output_format (str): Output format (html, pdf, markdown)
            
        Returns:
            str: Path to saved report
        """
        # Check if we have predictions
        if not predictions:
            if self.verbose:
                print("No predictions provided")
            return None
        
        # Get target IDs if not provided
        if target_ids is None:
            # Extract from predictions
            available_targets = set()
            for key, pred in predictions.items():
                if isinstance(pred, dict) and 'target_id' in pred:
                    available_targets.add(pred['target_id'])
            
            target_ids = sorted(available_targets)
            
            if not target_ids:
                # Fallback to targets in dataset
                if 'is_blue' in self.targets_df.columns:
                    targets_subset = self.targets_df[self.targets_df['is_blue'] == 0].copy()
                else:
                    targets_subset = self.targets_df.copy()
                
                target_ids = sorted(targets_subset['target_id'].unique())
        
        # Use provided time horizons or all available
        if time_horizons is None:
            # Find all available time horizons
            available_horizons = set()
            for key, pred in predictions.items():
                if isinstance(pred, dict) and 'minutes_ahead' in pred:
                    available_horizons.add(pred['minutes_ahead'])
            
            time_horizons = sorted(available_horizons)
        
        # Use provided methods or all available
        if methods is None:
            # Find all available methods
            available_methods = set()
            for key, pred in predictions.items():
                if isinstance(pred, dict) and 'method' in pred:
                    available_methods.add(pred['method'])
            
            methods = sorted(available_methods)
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        if output_format == 'html':
            filename = f"prediction_report_{timestamp}.html"
        elif output_format == 'pdf':
            filename = f"prediction_report_{timestamp}.pdf"
        else:  # markdown
            filename = f"prediction_report_{timestamp}.md"
        
        # Ensure output folder exists
        output_folder = self.config['output_folder']
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            filepath = os.path.join(output_folder, filename)
        else:
            filepath = filename
        
        # Generate visualizations
        visualizations = []
        
        # Create terrain map
        if not self.terrain_grid_df.empty:
            # Static terrain map
            terrain_fig = plt.figure(figsize=(self.config['fig_width'], self.config['fig_height']))
            terrain_ax = terrain_fig.add_subplot(111)
            self.plot_terrain(ax=terrain_ax)
            terrain_ax.set_title("Terrain and Land Use", fontsize=self.config['title_fontsize'])
            
            # Save figure
            terrain_path = os.path.join(output_folder, f"terrain_map_{timestamp}.png")
            terrain_fig.savefig(terrain_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close(terrain_fig)
            
            visualizations.append({
                'title': "Terrain and Land Use Map",
                'path': terrain_path,
                'type': 'image'
            })
        
        # Create target trajectory map
        trajectory_fig = plt.figure(figsize=(self.config['fig_width'], self.config['fig_height']))
        trajectory_ax = trajectory_fig.add_subplot(111)
        self.plot_target_trajectories(target_ids=target_ids, ax=trajectory_ax, 
                                     show_terrain=True, show_blue_forces=True)
        trajectory_ax.set_title("Target Trajectories", fontsize=self.config['title_fontsize'])
        
        # Save figure
        trajectory_path = os.path.join(output_folder, f"trajectories_{timestamp}.png")
        trajectory_fig.savefig(trajectory_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(trajectory_fig)
        
        visualizations.append({
            'title': "Target Trajectories",
            'path': trajectory_path,
            'type': 'image'
        })
        
        # Create interactive map
        interactive_map = self.create_multi_time_interactive_map(
            predictions, target_ids, time_horizons, methods[0] if methods else 'integrated'
        )
        
        # Save map
        map_path = os.path.join(output_folder, f"interactive_map_{timestamp}.html")
        interactive_map.save(map_path)
        
        visualizations.append({
            'title': "Interactive Prediction Map",
            'path': map_path,
            'type': 'html'
        })
        
        # Create prediction visualizations for each target
        for target_id in target_ids:
            # Multi-horizon prediction
            multi_horizon_preds = {}
            for key, pred in predictions.items():
                if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                    if 'method' in pred and pred['method'] == methods[0] if methods else 'integrated':
                        multi_horizon_preds[pred['minutes_ahead']] = pred
            
            if multi_horizon_preds:
                time_keys = sorted(multi_horizon_preds.keys())
                selected_times = time_keys[:min(4, len(time_keys))]
                
                # Create figure
                multi_fig = self.plot_multi_horizon_prediction(
                    {k: multi_horizon_preds[k] for k in selected_times},
                    target_id
                )
                
                if multi_fig:
                    # Save figure
                    multi_path = os.path.join(output_folder, f"target_{target_id}_multi_horizon_{timestamp}.png")
                    multi_fig.savefig(multi_path, dpi=self.config['dpi'], bbox_inches='tight')
                    plt.close(multi_fig)
                    
                    visualizations.append({
                        'title': f"Multi-Horizon Prediction for Target {target_id}",
                        'path': multi_path,
                        'type': 'image'
                    })
            
            # Method comparison
            if len(methods) > 1:
                method_preds = {}
                for method in methods:
                    for key, pred in predictions.items():
                        if (isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id and
                            'method' in pred and pred['method'] == method and
                            'minutes_ahead' in pred and pred['minutes_ahead'] == time_horizons[0] if time_horizons else 30):
                            method_preds[method] = pred
                
                if len(method_preds) > 1:
                    # Create figure
                    compare_fig = self.plot_prediction_comparison(
                        method_preds,
                        target_id
                    )
                    
                    if compare_fig:
                        # Save figure
                        compare_path = os.path.join(output_folder, f"target_{target_id}_methods_{timestamp}.png")
                        compare_fig.savefig(compare_path, dpi=self.config['dpi'], bbox_inches='tight')
                        plt.close(compare_fig)
                        
                        visualizations.append({
                            'title': f"Method Comparison for Target {target_id}",
                            'path': compare_path,
                            'type': 'image'
                        })
        
        # Create 3D terrain visualization
        if not self.terrain_grid_df.empty and 'elevation' in self.terrain_grid_df.columns:
            # Find a prediction to visualize
            target_id = target_ids[0] if target_ids else None
            viz_pred = None
            
            if target_id:
                for key, pred in predictions.items():
                    if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                        if 'minutes_ahead' in pred and pred['minutes_ahead'] == time_horizons[0] if time_horizons else 30:
                            if 'method' in pred and pred['method'] == methods[0] if methods else 'integrated':
                                viz_pred = pred
                                break
            
            # Create 3D visualization
            terrain_3d = self.create_3d_terrain_plot(
                viz_pred, target_id, interactive=True
            )
            
            if terrain_3d:
                # Save visualization
                terrain_3d_path = os.path.join(output_folder, f"terrain_3d_{timestamp}.html")
                terrain_3d.write_html(terrain_3d_path)
                
                visualizations.append({
                    'title': "3D Terrain Visualization",
                    'path': terrain_3d_path,
                    'type': 'html'
                })
        
        # Create tactical overlay
        tactical_fig = self.plot_military_overlay(
            predictions, target_ids, 
            time_horizons[0] if time_horizons else 30,
            methods[0] if methods else 'integrated'
        )
        
        if tactical_fig:
            # Save figure
            tactical_path = os.path.join(output_folder, f"tactical_overlay_{timestamp}.png")
            tactical_fig.savefig(tactical_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close(tactical_fig)
            
            visualizations.append({
                'title': "Tactical Situation Overlay",
                'path': tactical_path,
                'type': 'image'
            })
        
        # Generate report in selected format
        if output_format == 'html':
            self._generate_html_report(filepath, visualizations, target_ids, time_horizons, methods)
        elif output_format == 'pdf':
            self._generate_pdf_report(filepath, visualizations, target_ids, time_horizons, methods)
        else:  # markdown
            self._generate_markdown_report(filepath, visualizations, target_ids, time_horizons, methods)
        
        if self.verbose:
            print(f"Report generated successfully: {filepath}")
        
        return filepath
    
    def _generate_html_report(self, filepath, visualizations, target_ids, time_horizons, methods):
        """
        Generate HTML report with visualizations.
        
        Args:
            filepath (str): Output file path
            visualizations (list): List of visualization info dictionaries
            target_ids (list): List of target IDs
            time_horizons (list): List of time horizons
            methods (list): List of prediction methods
        """
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Predictive Target Tracking Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333366; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .viz {{ margin-bottom: 40px; }}
                .viz-image {{ text-align: center; max-width: 100%; }}
                .viz-image img {{ max-width: 100%; height: auto; }}
                .viz-html {{ width: 100%; height: 600px; border: 1px solid #ddd; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .footer {{ margin-top: 50px; text-align: center; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Predictive Target Tracking Report</h1>
                <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>This report contains visualizations of predicted target movements based on historical observations, terrain analysis, and tactical behavior modeling.</p>
                    
                    <h3>Targets</h3>
                    <ul>
        """
        
        # Add target information
        for target_id in target_ids:
            target_data = self.targets_df[self.targets_df['target_id'] == target_id]
            if not target_data.empty:
                target_class = target_data['target_class'].iloc[0] if 'target_class' in target_data.columns else 'unknown'
                last_seen = target_data['timestamp'].max().strftime(self.config['timestamp_format']) if 'timestamp' in target_data.columns else 'unknown'
                html_content += f"<li><strong>Target {target_id}</strong> ({target_class}) - Last seen: {last_seen}</li>\n"
        
        html_content += """
                    </ul>
                    
                    <h3>Prediction Parameters</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Values</th>
                        </tr>
        """
        
        # Add time horizons
        time_str = ", ".join([f"{t} minutes" for t in time_horizons])
        html_content += f"""
                        <tr>
                            <td>Time Horizons</td>
                            <td>{time_str}</td>
                        </tr>
        """
        
        # Add methods
        method_str = ", ".join(methods)
        html_content += f"""
                        <tr>
                            <td>Prediction Methods</td>
                            <td>{method_str}</td>
                        </tr>
        """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Add visualizations
        for viz in visualizations:
            html_content += f"""
                <div class="viz">
                    <h2>{viz['title']}</h2>
            """
            
            if viz['type'] == 'image':
                # Get relative path
                rel_path = os.path.basename(viz['path'])
                html_content += f"""
                    <div class="viz-image">
                        <img src="{rel_path}" alt="{viz['title']}">
                    </div>
                """
            elif viz['type'] == 'html':
                # Get relative path
                rel_path = os.path.basename(viz['path'])
                html_content += f"""
                    <iframe class="viz-html" src="{rel_path}"></iframe>
                """
            
            html_content += """
                </div>
            """
        
        # Add footer
        html_content += """
                <div class="footer">
                    <p>Generated by the Predictive Target Tracking System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_pdf_report(self, filepath, visualizations, target_ids, time_horizons, methods):
        """
        Generate PDF report with visualizations.
        
        Args:
            filepath (str): Output file path
            visualizations (list): List of visualization info dictionaries
            target_ids (list): List of target IDs
            time_horizons (list): List of time horizons
            methods (list): List of prediction methods
        """
        try:
            from reportlab.pdfgen import canvas # type: ignore
            from reportlab.lib.pagesizes import letter # type: ignore
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle # type: ignore
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore
            from reportlab.lib import colors # type: ignore
            from reportlab.lib.units import inch # type: ignore
        except ImportError:
            if self.verbose:
                print("ReportLab not installed. Please install with: pip install reportlab")
                print("Falling back to Markdown report")
            
            # Fall back to markdown
            return self._generate_markdown_report(filepath.replace('.pdf', '.md'), 
                                                visualizations, target_ids, time_horizons, methods)
        
        # Create document
        doc = SimpleDocTemplate(filepath, pagesize=letter, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)
        
        # Get styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Title', fontSize=18, spaceAfter=12))
        styles.add(ParagraphStyle(name='Heading2', fontSize=14, spaceAfter=6))
        styles.add(ParagraphStyle(name='Heading3', fontSize=12, spaceAfter=6))
        styles.add(ParagraphStyle(name='Normal', fontSize=10, spaceAfter=6))
        
        # Create document elements
        elements = []
        
        # Title
        elements.append(Paragraph("Predictive Target Tracking Report", styles['Title']))
        elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary
        elements.append(Paragraph("Summary", styles['Heading2']))
        elements.append(Paragraph(
            "This report contains visualizations of predicted target movements based on historical observations, terrain analysis, and tactical behavior modeling.",
            styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Targets
        elements.append(Paragraph("Targets", styles['Heading3']))
        
        target_data = []
        target_data.append(["Target ID", "Class", "Last Seen"])
        
        for target_id in target_ids:
            target_subset = self.targets_df[self.targets_df['target_id'] == target_id]
            if not target_subset.empty:
                target_class = target_subset['target_class'].iloc[0] if 'target_class' in target_subset.columns else 'unknown'
                last_seen = target_subset['timestamp'].max().strftime(self.config['timestamp_format']) if 'timestamp' in target_subset.columns else 'unknown'
                target_data.append([str(target_id), target_class, last_seen])
        
        table = Table(target_data, colWidths=[1*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.1*inch))
        
        # Prediction Parameters
        elements.append(Paragraph("Prediction Parameters", styles['Heading3']))
        
        param_data = []
        param_data.append(["Parameter", "Values"])
        param_data.append(["Time Horizons", ", ".join([f"{t} minutes" for t in time_horizons])])
        param_data.append(["Prediction Methods", ", ".join(methods)])
        
        table = Table(param_data, colWidths=[1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Visualizations
        for viz in visualizations:
            if viz['type'] == 'image':
                elements.append(Paragraph(viz['title'], styles['Heading2']))
                elements.append(Spacer(1, 0.1*inch))
                
                # Add image
                img = Image(viz['path'], width=6*inch, height=5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
        
        # Add note about HTML visualizations
        html_viz = [viz for viz in visualizations if viz['type'] == 'html']
        if html_viz:
            elements.append(Paragraph("Interactive Visualizations", styles['Heading2']))
            elements.append(Paragraph(
                "The following interactive visualizations are available as HTML files:",
                styles['Normal']))
            
            for viz in html_viz:
                elements.append(Paragraph(f"• {viz['title']}: {os.path.basename(viz['path'])}", styles['Normal']))
        
        # Build PDF
        doc.build(elements)
    
    def _generate_markdown_report(self, filepath, visualizations, target_ids, time_horizons, methods):
        """
        Generate Markdown report with visualizations.
        
        Args:
            filepath (str): Output file path
            visualizations (list): List of visualization info dictionaries
            target_ids (list): List of target IDs
            time_horizons (list): List of time horizons
            methods (list): List of prediction methods
        """
        # Create markdown content
        md_content = f"""# Predictive Target Tracking Report

Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

This report contains visualizations of predicted target movements based on historical observations, terrain analysis, and tactical behavior modeling.

### Targets

| Target ID | Class | Last Seen |
|-----------|-------|-----------|
"""
        
        # Add target information
        for target_id in target_ids:
            target_data = self.targets_df[self.targets_df['target_id'] == target_id]
            if not target_data.empty:
                target_class = target_data['target_class'].iloc[0] if 'target_class' in target_data.columns else 'unknown'
                last_seen = target_data['timestamp'].max().strftime(self.config['timestamp_format']) if 'timestamp' in target_data.columns else 'unknown'
                md_content += f"| {target_id} | {target_class} | {last_seen} |\n"
        
        md_content += """
### Prediction Parameters

| Parameter | Values |
|-----------|--------|
"""
        
        # Add time horizons
        time_str = ", ".join([f"{t} minutes" for t in time_horizons])
        md_content += f"| Time Horizons | {time_str} |\n"
        
        # Add methods
        method_str = ", ".join(methods)
        md_content += f"| Prediction Methods | {method_str} |\n\n"
        
        # Add visualizations
        for viz in visualizations:
            md_content += f"## {viz['title']}\n\n"
            
            if viz['type'] == 'image':
                # Get relative path
                rel_path = os.path.basename(viz['path'])
                md_content += f"![{viz['title']}]({rel_path})\n\n"
            elif viz['type'] == 'html':
                # Get relative path
                rel_path = os.path.basename(viz['path'])
                md_content += f"[Interactive visualization: {viz['title']}]({rel_path})\n\n"
        
        # Add footer
        md_content += """
---

*Generated by the Predictive Target Tracking System*
"""
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineering
    from movement_predictor import MovementPredictor
    
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
    
    # Create movement predictor
    predictor = MovementPredictor(
        targets, 
        blue_forces, 
        weighted_cost, 
        resolution,
        verbose=True
    )
    
    # Generate predictions for a target
    target_id = targets['target_id'].iloc[0]
    
    # Generate heatmap prediction
    heatmap_30min = predictor.generate_probability_heatmap(target_id, 30, 'integrated')
    heatmap_60min = predictor.generate_probability_heatmap(target_id, 60, 'integrated')
    
    # Create visualizations
    viz = Visualization(targets, blue_forces, weighted_cost, verbose=True)
    
    # Plot 2D heatmap
    fig = viz.plot_prediction_heatmap(heatmap_30min)
    plt.savefig("prediction_heatmap.png", dpi=150, bbox_inches='tight')
    
    # Create interactive map
    m = viz.create_interactive_map(heatmap_30min)
    m.save("interactive_map.html")
    
    # Create 3D terrain plot
    terrain_3d = viz.create_3d_terrain_plot(heatmap_30min, target_id)
    
    # Create multi-horizon visualization
    multi_fig = viz.plot_multi_horizon_prediction(
        {30: heatmap_30min, 60: heatmap_60min},
        target_id
    )
    plt.savefig("multi_horizon.png", dpi=150, bbox_inches='tight')
    
    # Generate comprehensive report
    predictions = {
        "30min": heatmap_30min,
        "60min": heatmap_60min
    }
    
    report_path = viz.generate_report(predictions, output_format='html')
    
    print(f"Visualization complete. Report generated: {report_path}")