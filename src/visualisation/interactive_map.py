"""
Interactive Map Module

This module provides specialized interactive mapping capabilities for the
predictive tracking system, with a focus on web-based visualizations and
advanced geospatial analysis.

Key capabilities:
- Interactive web maps with multiple data layers
- Time-based animation of target movements and predictions
- Choropleth maps for terrain cost analysis
- Interactive 3D terrain visualization
- Military symbology overlay
- Comparative heatmap analysis
- Geospatial analytics dashboard generation
"""

import folium # type: ignore
from folium.plugins import ( # type: ignore
    HeatMap, MarkerCluster, MeasureControl, TimestampedGeoJson,
    MousePosition, Draw, Fullscreen, MiniMap, FloatImage,
    DualMap, HeatMapWithTime, TimeDimension, AntPath
)
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import branca # type: ignore
import branca.colormap as cm # type: ignore
from shapely.geometry import Point, LineString, Polygon # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
import geopandas as gpd # type: ignore
import json
import datetime
import os
import base64
from io import BytesIO
import math
import tempfile
import warnings
warnings.filterwarnings('ignore')

class InteractiveMap:
    """
    Class for creating advanced interactive maps for predictive tracking visualizations.
    
    This class extends the basic visualization capabilities with specialized
    interactive mapping features.
    
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
        Initialize the InteractiveMap class.
        
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
            'map_style': 'openstreetmap',  # Base map style
            'map_tiles': {
                'openstreetmap': 'OpenStreetMap',
                'terrain': 'Stamen Terrain',
                'satellite': 'Esri Satellite',
                'dark': 'CartoDB dark_matter',
                'light': 'CartoDB positron',
                'topo': 'Stamen Topo'
            },
            'heatmap_colors': ['blue', 'cyan', 'green', 'yellow', 'red'],  # Heatmap color gradient
            'target_colors': {  # Colors for different target classes
                'vehicle': '#FF0000',    # Red
                'infantry': '#008000',   # Green
                'artillery': '#FFA500',  # Orange
                'command': '#0000FF'     # Blue
            },
            'blue_force_color': '#0000FF',  # Blue
            'prediction_opacity': 0.7,  # Opacity of prediction overlays
            'output_folder': './output',  # Folder for saving visualizations
            'dpi': 150,  # DPI for saved figures
            'military_symbols': True,  # Whether to use military symbology
            'timestamp_format': '%Y-%m-%d %H:%M',  # Format for timestamps
            'popup_width': 300,  # Width of popups in pixels
            'enable_draw': True,  # Enable drawing tools
            'enable_fullscreen': True,  # Enable fullscreen control
            'enable_measure': True,  # Enable measurement control
            'enable_mini_map': True,  # Enable mini map
            'enable_mouse_position': True,  # Enable mouse position display
            'map_width': '100%',  # Map width (CSS value)
            'map_height': '700px',  # Map height (CSS value)
            'animation_duration': 10,  # Animation duration in seconds
            'animation_fps': 10,  # Animation frames per second
            'zoom_start': 12,  # Initial zoom level
            'geo_json_style': {  # Default GeoJSON style
                'color': 'black',
                'weight': 2,
                'opacity': 0.7,
                'fillOpacity': 0.5
            }
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Create output folder if it doesn't exist
        if self.config['output_folder']:
            os.makedirs(self.config['output_folder'], exist_ok=True)
        
        # Create colormap
        self.heatmap_colormap = LinearSegmentedColormap.from_list('custom_heatmap', 
            self.config['heatmap_colors'], N=256)
        
        if self.verbose:
            print("Interactive Map module initialized")
    
    def create_base_map(self, center=None, zoom_start=None, width=None, height=None, 
                        tiles=None, attr=None, control_scale=True):
        """
        Create a base map with optional controls.
        
        Args:
            center (tuple): Center coordinates (lat, lon) (optional)
            zoom_start (int): Initial zoom level (optional)
            width (str): Width of the map (optional)
            height (str): Height of the map (optional)
            tiles (str): Tile layer name (optional)
            attr (str): Attribution for custom tiles (optional)
            control_scale (bool): Whether to add a scale control
            
        Returns:
            folium.Map: Base map
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
        
        # Get configuration parameters or defaults
        zoom_start = zoom_start if zoom_start is not None else self.config['zoom_start']
        width = width if width is not None else self.config['map_width']
        height = height if height is not None else self.config['map_height']
        tiles = tiles if tiles is not None else self.config['map_style']
        
        # Get tile layer name if provided as key
        if tiles in self.config['map_tiles']:
            tiles = self.config['map_tiles'][tiles]
        
        # Create map
        m = folium.Map(location=center, 
                      zoom_start=zoom_start, 
                      tiles=tiles,
                      attr=attr,
                      width=width, 
                      height=height,
                      control_scale=control_scale)
        
        # Add additional base tile layers for quick switching
        for tile_key, tile_name in self.config['map_tiles'].items():
            if tile_name != tiles:  # Skip the default tile we already added
                folium.TileLayer(tile_name).add_to(m)
        
        # Add controls if configured
        if self.config['enable_measure']:
            m.add_child(MeasureControl())
        
        if self.config['enable_fullscreen']:
            Fullscreen().add_to(m)
        
        if self.config['enable_draw']:
            Draw(export=True, filename='drawn_features.geojson').add_to(m)
        
        if self.config['enable_mouse_position']:
            MousePosition().add_to(m)
        
        if self.config['enable_mini_map']:
            MiniMap(toggle_display=True).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def add_target_trails(self, m, target_ids=None, show_markers=True, show_labels=True, 
                          animate=False, layer_name="Target Tracks"):
        """
        Add target trails to the map.
        
        Args:
            m (folium.Map): Map to add trails to
            target_ids (list): List of target IDs to include (optional)
            show_markers (bool): Whether to show markers
            show_labels (bool): Whether to show labels
            animate (bool): Whether to animate the trails over time
            layer_name (str): Name for the layer group
            
        Returns:
            folium.Map: Map with target trails
        """
        # Create feature group for this layer
        target_group = folium.FeatureGroup(name=layer_name)
        
        # Filter targets if target_ids provided
        if target_ids is not None:
            targets_subset = self.targets_df[self.targets_df['target_id'].isin(target_ids)].copy()
        else:
            targets_subset = self.targets_df.copy()
        
        # Filter out blue forces from targets
        if 'is_blue' in targets_subset.columns:
            targets_subset = targets_subset[targets_subset['is_blue'] == 0].copy()
        
        # Get unique target IDs
        unique_targets = sorted(targets_subset['target_id'].unique())
        
        if animate:
            # For animation, we'll use TimestampedGeoJson
            # Prepare features list
            features = []
            
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
                
                # Create features for each point
                for idx, row in target_data.iterrows():
                    # Convert timestamp to Unix time (milliseconds)
                    timestamp = int(row['timestamp'].timestamp() * 1000)
                    
                    # Create feature for point
                    point_feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [row['longitude'], row['latitude']]
                        },
                        'properties': {
                            'time': timestamp,
                            'icon': 'circle',
                            'iconstyle': {
                                'fillColor': color,
                                'fillOpacity': 0.8,
                                'stroke': 'true',
                                'radius': 5
                            },
                            'style': {'weight': 0},
                            'popup': f"Target {target_id} ({target_class})<br>Time: {row['timestamp'].strftime(self.config['timestamp_format'])}"
                        }
                    }
                    features.append(point_feature)
                
                # Create line features connecting consecutive points
                for i in range(len(target_data) - 1):
                    point1 = target_data.iloc[i]
                    point2 = target_data.iloc[i + 1]
                    
                    # Use the later timestamp for the line
                    timestamp = int(point2['timestamp'].timestamp() * 1000)
                    
                    line_feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': [
                                [point1['longitude'], point1['latitude']],
                                [point2['longitude'], point2['latitude']]
                            ]
                        },
                        'properties': {
                            'time': timestamp,
                            'style': {
                                'color': color,
                                'weight': 3,
                                'opacity': 0.8
                            }
                        }
                    }
                    features.append(line_feature)
            
            # Create TimestampedGeoJson layer
            time_options = {
                'period': 'PT1M',  # 1 minute per animation step
                'duration': f'PT{self.config["animation_duration"]}S',  # Animation duration
                'speed': 10,  # Animation speed
                'startTime': min(target_data['timestamp']).strftime('%Y-%m-%dT%H:%M:%S'),
                'endTime': max(target_data['timestamp']).strftime('%Y-%m-%dT%H:%M:%S'),
                'timeFormat': 'iso8601'
            }
            
            TimestampedGeoJson(
                {
                    'type': 'FeatureCollection',
                    'features': features
                },
                period='PT1M',
                duration=f'PT{self.config["animation_duration"]}S',
                transition_time=self.config["animation_duration"] * 1000 / len(features),
                auto_play=True,
                loop=True,
                max_speed=1,
                loop_button=True,
                date_options='YYYY-MM-DD HH:mm',
                time_slider_drag_update=True
            ).add_to(target_group)
                
        else:
            # For static display, add polylines and markers
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
                
                # Create trail as an animated path
                points = list(zip(target_data['latitude'], target_data['longitude']))
                if len(points) >= 2:
                    # Use AntPath for animated trails (looks like ants marching)
                    AntPath(
                        locations=points,
                        color=color,
                        weight=3,
                        opacity=0.8,
                        dash_array=[10, 20],
                        delay=1000,
                        pulse_color='#FFFFFF',
                        pulse_opacity=0.5,
                        tooltip=f"Target {target_id} ({target_class})"
                    ).add_to(target_group)
                
                # Add markers if requested
                if show_markers:
                    # Add points along the trail
                    for idx, row in target_data.iterrows():
                        popup_html = f"""
                        <div style="width: {self.config['popup_width']}px">
                        <h4>Target {target_id}</h4>
                        <b>Class:</b> {target_class}<br>
                        <b>Time:</b> {row['timestamp'].strftime(self.config['timestamp_format'])}<br>
                        """
                        
                        if 'speed' in row:
                            popup_html += f"<b>Speed:</b> {row['speed']:.1f} km/h<br>"
                        
                        if 'heading' in row:
                            popup_html += f"<b>Heading:</b> {row['heading']:.1f}°<br>"
                        
                        popup_html += "</div>"
                        
                        # Get index in the sequence for marker color
                        idx_norm = idx / max(1, len(target_data) - 1)  # 0 to 1
                        
                        # Create a circular marker with color showing sequence
                        icon_color = self._blend_colors('blue', color, idx_norm)
                        
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=5,
                            color=icon_color,
                            fill=True,
                            fill_color=icon_color,
                            fill_opacity=0.8,
                            popup=folium.Popup(popup_html, max_width=self.config['popup_width']),
                            tooltip=f"Target {target_id} - {row['timestamp'].strftime('%H:%M')}"
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
                    last_row = target_data.iloc[-1]
                    
                    # Create popup with target information
                    popup_html = f"""
                    <div style="width: {self.config['popup_width']}px">
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
                    
                    # Add label if requested
                    if show_labels:
                        folium.Marker(
                            end_point,
                            icon=folium.DivIcon(
                                icon_size=(150, 36),
                                icon_anchor=(75, 18),
                                html=f'<div style="font-size: 12pt; color: {color}; text-align: center; text-shadow: 0 0 3px white;"><b>Target {target_id}</b></div>'
                            )
                        ).add_to(target_group)
                    
                    # Create marker
                    if 'heading' in last_row:
                        # Use custom icon with heading indicator
                        icon_html = self._create_heading_icon(color, last_row['heading'])
                        icon = folium.DivIcon(
                            icon_size=(30, 30),
                            icon_anchor=(15, 15),
                            html=icon_html
                        )
                    else:
                        # Use standard icon
                        icon = folium.Icon(color='red', icon='stop')
                    
                    folium.Marker(
                        end_point, 
                        popup=folium.Popup(popup_html, max_width=self.config['popup_width']),
                        icon=icon,
                        tooltip=f"Target {target_id} last known position"
                    ).add_to(target_group)
        
        # Add layer to map
        target_group.add_to(m)
        
        return m
    
    def add_blue_forces(self, m, show_labels=True, layer_name="Blue Forces"):
        """
        Add blue forces to the map.
        
        Args:
            m (folium.Map): Map to add blue forces to
            show_labels (bool): Whether to show labels
            layer_name (str): Name for the layer group
            
        Returns:
            folium.Map: Map with blue forces
        """
        # Skip if no blue forces
        if self.blue_forces_df.empty:
            return m
        
        # Create feature group for this layer
        blue_group = folium.FeatureGroup(name=layer_name)
        
        # Get most recent position for each blue force
        if 'timestamp' in self.blue_forces_df.columns:
            blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
        else:
            blue_latest = self.blue_forces_df.copy()
        
        # Create marker cluster for blue forces
        blue_cluster = MarkerCluster(name="Blue Forces").add_to(blue_group)
        
        # Add markers for each blue force
        for idx, row in blue_latest.iterrows():
            # Get blue force class
            blue_class = row['blue_class'] if 'blue_class' in row else 'unknown'
            
            # Create popup with blue force information
            popup_html = f"""
            <div style="width: {self.config['popup_width']}px">
            <h4>Blue Force {row['blue_id']}</h4>
            <b>Class:</b> {blue_class}<br>
            """
            
            if 'timestamp' in row:
                popup_html += f"<b>Last update:</b> {row['timestamp'].strftime(self.config['timestamp_format'])}<br>"
            
            popup_html += "</div>"
            
            # Add label if requested
            if show_labels:
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    icon=folium.DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(75, 18),
                        html=f'<div style="font-size: 12pt; color: blue; text-align: center; text-shadow: 0 0 3px white;"><b>Blue {row["blue_id"]}</b></div>'
                    )
                ).add_to(blue_group)
            
            # Create marker
            if 'heading' in row:
                # Use custom icon with heading indicator
                icon_html = self._create_heading_icon('blue', row['heading'])
                icon = folium.DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html=icon_html
                )
            else:
                # Use standard icon
                icon = folium.Icon(color='blue', icon='info-sign')
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=self.config['popup_width']),
                icon=icon,
                tooltip=f"Blue Force {row['blue_id']}"
            ).add_to(blue_cluster)
        
        # Add layer to map
        blue_group.add_to(m)
        
        return m
    
    def add_prediction_heatmap(self, m, prediction_data, target_id=None, time_horizon=None, method=None,
                               radius=15, blur=10, layer_name=None):
        """
        Add prediction heatmap to the map.
        
        Args:
            m (folium.Map): Map to add heatmap to
            prediction_data (dict): Prediction data
            target_id: Target ID (optional, only used for layer name)
            time_horizon: Time horizon in minutes (optional, only used for layer name)
            method: Prediction method (optional, only used for layer name)
            radius (int): Radius of heatmap points
            blur (int): Blur radius for heatmap
            layer_name (str): Name for the layer group (optional)
            
        Returns:
            folium.Map: Map with prediction heatmap
        """
        if prediction_data is None or 'density' not in prediction_data:
            if self.verbose:
                print("No prediction data provided")
            return m
        
        # Extract prediction data
        lat_grid = prediction_data['lat_grid']
        lon_grid = prediction_data['lon_grid']
        density = prediction_data['density']
        
        # Get metadata if available
        if target_id is None and 'target_id' in prediction_data:
            target_id = prediction_data['target_id']
        
        if time_horizon is None and 'minutes_ahead' in prediction_data:
            time_horizon = prediction_data['minutes_ahead']
        
        if method is None and 'method' in prediction_data:
            method = prediction_data['method']
        
        # Create layer name if not provided
        if layer_name is None:
            if target_id is not None and time_horizon is not None:
                if method is not None:
                    layer_name = f"Target {target_id} - T+{time_horizon}min ({method})"
                else:
                    layer_name = f"Target {target_id} - T+{time_horizon}min"
            else:
                layer_name = "Prediction Heatmap"
        
        # Create feature group for this layer
        heatmap_group = folium.FeatureGroup(name=layer_name)
        
        # Convert to points for heatmap
        points = []
        intensities = []
        
        # Sample points from the density grid
        for i in range(len(lat_grid)):
            for j in range(len(lon_grid)):
                if density[i, j] > 0.01:  # Skip very low probability areas
                    points.append([lat_grid[i], lon_grid[j]])
                    intensities.append(float(density[i, j]))
        
        # Add heatmap
        HeatMap(
            points,
            name=layer_name,
            min_opacity=0.2,
            max_zoom=18,
            radius=radius,
            blur=blur,
            gradient={0.4: 'blue', 0.65: 'lime', 0.9: 'yellow', 1: 'red'},
            max_val=max(intensities) if intensities else 1.0,
            overlay=True,
            control=True,
            intensity=intensities
        ).add_to(heatmap_group)
        
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
                        
                        # Determine color for this confidence level
                        # Normalize level to 0-1 range
                        norm_level = level / 100
                        
                        if norm_level <= 0.7:
                            color = 'red'
                        elif norm_level <= 0.9:
                            color = 'orange'
                        else:
                            color = 'blue'
                        
                        # Add polygon
                        folium.Polygon(
                            locations=hull_points,
                            color=color,
                            weight=1,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.2,
                            tooltip=f"{level}% Confidence Region"
                        ).add_to(heatmap_group)
                    except:
                        # If convex hull fails, skip
                        pass
        
        # Add most likely position marker
        if target_id is not None:
            # Find maximum density point
            max_idx = np.argmax(density)
            max_i, max_j = np.unravel_index(max_idx, density.shape)
            max_lat = lat_grid[max_i]
            max_lon = lon_grid[max_j]
            
            # Get target class for color if available
            target_class = prediction_data.get('target_class', 'unknown')
            color = self.config['target_colors'].get(target_class, 'red')
            
            # Create popup with prediction information
            popup_html = f"""
            <div style="width: {self.config['popup_width']}px">
            <h4>Target {target_id} Prediction</h4>
            <b>Time horizon:</b> {time_horizon} minutes<br>
            """
            
            if method is not None:
                popup_html += f"<b>Method:</b> {method}<br>"
            
            if 'confidence' in prediction_data:
                confidence = prediction_data['confidence'] * 100
                popup_html += f"<b>Confidence:</b> {confidence:.1f}%<br>"
            
            popup_html += f"<b>Class:</b> {target_class}<br>"
            
            # Add confidence region areas if available
            if 'confidence_regions' in prediction_data:
                popup_html += "<b>Confidence regions:</b><br>"
                for level, region in prediction_data['confidence_regions'].items():
                    if 'area_km2' in region:
                        popup_html += f"&nbsp;&nbsp;{level}%: {region['area_km2']:.2f} km²<br>"
            
            popup_html += "</div>"
            
            # Add marker for most likely position
            icon = folium.Icon(color=self._color_to_name(color), icon='crosshairs')
            
            folium.Marker(
                [max_lat, max_lon],
                popup=folium.Popup(popup_html, max_width=self.config['popup_width']),
                icon=icon,
                tooltip=f"Target {target_id} - Most likely position at T+{time_horizon}min"
            ).add_to(heatmap_group)
        
        # Add layer to map
        heatmap_group.add_to(m)
        
        return m
    
    def add_terrain_overlay(self, m, layer_name="Terrain", feature_type="elevation"):
        """
        Add terrain overlay to the map.
        
        Args:
            m (folium.Map): Map to add terrain overlay to
            layer_name (str): Name for the layer group
            feature_type (str): Terrain feature to display ('elevation', 'cost', 'concealment')
            
        Returns:
            folium.Map: Map with terrain overlay
        """
        if self.terrain_grid_df.empty:
            if self.verbose:
                print("No terrain data available")
            return m
        
        # Check if we have the requested feature
        if feature_type == 'elevation' and 'elevation' not in self.terrain_grid_df.columns:
            if self.verbose:
                print("No elevation data available")
            return m
        
        if feature_type == 'cost' and 'total_cost' not in self.terrain_grid_df.columns:
            if self.verbose:
                print("No terrain cost data available")
            return m
        
        if feature_type == 'concealment' and 'concealment' not in self.terrain_grid_df.columns:
            if self.verbose:
                print("No concealment data available")
            return m
        
        # Create feature group for this layer
        terrain_group = folium.FeatureGroup(name=layer_name)
        
        # Determine color scale based on feature type
        if feature_type == 'elevation':
            feature_col = 'elevation'
            min_val = self.terrain_grid_df['elevation'].min()
            max_val = self.terrain_grid_df['elevation'].max()
            colormap = cm.LinearColormap(
                ['green', 'yellow', 'brown', 'white'],
                vmin=min_val, vmax=max_val,
                caption='Elevation (m)'
            )
        elif feature_type == 'cost':
            feature_col = 'total_cost'
            min_val = self.terrain_grid_df['total_cost'].min()
            max_val = self.terrain_grid_df['total_cost'].max()
            colormap = cm.LinearColormap(
                ['green', 'yellow', 'orange', 'red'],
                vmin=min_val, vmax=max_val,
                caption='Terrain Cost'
            )
        elif feature_type == 'concealment':
            feature_col = 'concealment'
            min_val = 0
            max_val = 1
            colormap = cm.LinearColormap(
                ['red', 'yellow', 'green'],
                vmin=min_val, vmax=max_val,
                caption='Concealment'
            )
        else:
            if self.verbose:
                print(f"Unknown feature type: {feature_type}")
            return m
        
        # Create grid for interpolation
        grid_size = int(np.sqrt(len(self.terrain_grid_df)))
        
        # Try to reshape the data to grid
        try:
            lat_grid = self.terrain_grid_df['latitude'].values.reshape(grid_size, grid_size)
            lon_grid = self.terrain_grid_df['longitude'].values.reshape(grid_size, grid_size)
            feature_grid = self.terrain_grid_df[feature_col].values.reshape(grid_size, grid_size)
            
            # Create polygons for each grid cell
            for i in range(grid_size - 1):
                for j in range(grid_size - 1):
                    # Define polygon coordinates
                    polygon = [
                        [lat_grid[i, j], lon_grid[i, j]],
                        [lat_grid[i+1, j], lon_grid[i+1, j]],
                        [lat_grid[i+1, j+1], lon_grid[i+1, j+1]],
                        [lat_grid[i, j+1], lon_grid[i, j+1]]
                    ]
                    
                    # Calculate average feature value for this cell
                    avg_val = (feature_grid[i, j] + feature_grid[i+1, j] + 
                              feature_grid[i+1, j+1] + feature_grid[i, j+1]) / 4
                    
                    # Get color for this value
                    color = colormap(avg_val)
                    
                    # Create polygon with this color
                    folium.Polygon(
                        locations=polygon,
                        color=None,
                        weight=0,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5,
                        tooltip=f"{feature_type.capitalize()}: {avg_val:.2f}"
                    ).add_to(terrain_group)
        except:
            # If reshaping fails, use individual points
            if self.verbose:
                print("Failed to reshape terrain data, using points instead")
            
            # Create circles for each point
            for idx, row in self.terrain_grid_df.iterrows():
                # Get color for this value
                color = colormap(row[feature_col])
                
                # Create circle with this color
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color=None,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.5,
                    tooltip=f"{feature_type.capitalize()}: {row[feature_col]:.2f}"
                ).add_to(terrain_group)
        
        # Add colormap to map
        colormap.add_to(m)
        
        # Add layer to map
        terrain_group.add_to(m)
        
        return m
    
    def add_land_use(self, m, layer_name="Land Use"):
        """
        Add land use overlay to the map.
        
        Args:
            m (folium.Map): Map to add land use overlay to
            layer_name (str): Name for the layer group
            
        Returns:
            folium.Map: Map with land use overlay
        """
        if self.terrain_grid_df.empty or 'land_use_type' not in self.terrain_grid_df.columns:
            if self.verbose:
                print("No land use data available")
            return m
        
        # Create feature group for this layer
        landuse_group = folium.FeatureGroup(name=layer_name)
        
        # Get unique land use types
        land_use_types = self.terrain_grid_df['land_use_type'].unique()
        
        # Define colors for land use types
        land_use_colors = {
            'urban': '#A9A9A9',     # Dark gray
            'road': '#000000',      # Black
            'forest': '#228B22',    # Forest green
            'open': '#90EE90',      # Light green
            'water': '#1E90FF',     # Dodger blue
            'wetland': '#7D9EC0',   # Light slate
            'restricted': '#FF8C00' # Dark orange
        }
        
        # Group by land use type and create polygons
        for land_use_type in land_use_types:
            # Get points for this land use type
            type_points = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == land_use_type]
            
            # Get color for this type
            color = land_use_colors.get(land_use_type, '#CCCCCC')
            
            # Try to create polygons for each land use area
            try:
                # Create points
                points = [Point(row['longitude'], row['latitude']) for _, row in type_points.iterrows()]
                
                # Buffer points to create areas
                areas = [p.buffer(0.001) for p in points]
                
                # Combine overlapping areas
                from shapely.ops import unary_union # type: ignore
                combined_areas = unary_union(areas)
                
                # Add to map
                if hasattr(combined_areas, 'geoms'):
                    # Multiple polygons
                    for poly in combined_areas.geoms:
                        # Convert to list of coordinates
                        coords = [(y, x) for x, y in zip(*poly.exterior.xy)]
                        
                        # Create polygon
                        folium.Polygon(
                            locations=coords,
                            color=color,
                            weight=1,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.5,
                            tooltip=f"Land use: {land_use_type}"
                        ).add_to(landuse_group)
                else:
                    # Single polygon
                    coords = [(y, x) for x, y in zip(*combined_areas.exterior.xy)]
                    
                    # Create polygon
                    folium.Polygon(
                        locations=coords,
                        color=color,
                        weight=1,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5,
                        tooltip=f"Land use: {land_use_type}"
                    ).add_to(landuse_group)
            except:
                # If combining fails, use individual points
                for idx, row in type_points.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color=None,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5,
                        tooltip=f"Land use: {land_use_type}"
                    ).add_to(landuse_group)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; background-color: white; 
                    padding: 10px; border-radius: 5px; z-index: 1000; max-width: 250px;">
        <h4>Land Use Legend</h4>
        <table>
        """
        
        for land_use, color in land_use_colors.items():
            if land_use in land_use_types:
                legend_html += f"""
                <tr>
                    <td style="width: 20px; height: 20px; background-color: {color};"></td>
                    <td style="padding-left: 10px;">{land_use.capitalize()}</td>
                </tr>
                """
        
        legend_html += """
        </table>
        </div>
        """
        
        # Add legend as a FloatImage
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer to map
        landuse_group.add_to(m)
        
        return m
    
    def add_mobility_network(self, m, mobility_network, layer_name="Mobility Network", show_costs=True):
        """
        Add mobility network to the map.
        
        Args:
            m (folium.Map): Map to add mobility network to
            mobility_network (nx.Graph): NetworkX graph of mobility network
            layer_name (str): Name for the layer group
            show_costs (bool): Whether to show edge costs as weights
            
        Returns:
            folium.Map: Map with mobility network
        """
        if mobility_network is None:
            if self.verbose:
                print("No mobility network provided")
            return m
        
        # Create feature group for this layer
        network_group = folium.FeatureGroup(name=layer_name)
        
        # Add edges
        for u, v, data in mobility_network.edges(data=True):
            # Get node positions
            if 'pos' in mobility_network.nodes[u] and 'pos' in mobility_network.nodes[v]:
                u_pos = mobility_network.nodes[u]['pos']
                v_pos = mobility_network.nodes[v]['pos']
                
                # Get edge weight
                weight = data.get('weight', 1.0)
                
                # Scale weight for display (thicker = less cost)
                display_weight = max(1, int(5 / (weight + 0.1)))
                
                # Determine color based on weight
                if weight < 1.5:
                    color = 'green'
                elif weight < 3:
                    color = 'yellow'
                elif weight < 5:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Create line
                folium.PolyLine(
                    locations=[[u_pos[1], u_pos[0]], [v_pos[1], v_pos[0]]],
                    color=color,
                    weight=display_weight,
                    opacity=0.8,
                    tooltip=f"Cost: {weight:.2f}" if show_costs else None
                ).add_to(network_group)
        
        # Add nodes
        for node, data in mobility_network.nodes(data=True):
            if 'pos' in data:
                pos = data['pos']
                
                # Get node cost if available
                cost = data.get('cost', None)
                
                # Determine radius based on node importance
                radius = 3
                
                # Create circle
                folium.CircleMarker(
                    location=[pos[1], pos[0]],
                    radius=radius,
                    color='black',
                    fill=True,
                    fill_opacity=0.5,
                    tooltip=f"Node {node}, Cost: {cost:.2f}" if cost is not None else f"Node {node}"
                ).add_to(network_group)
        
        # Add layer to map
        network_group.add_to(m)
        
        return m
    
    def add_time_horizon_control(self, m, predictions, target_id, methods=None):
        """
        Add a control to switch between time horizons.
        
        Args:
            m (folium.Map): Map to add control to
            predictions (dict): Dictionary of prediction results
            target_id: Target ID to show predictions for
            methods (list): List of methods to include (optional)
            
        Returns:
            folium.Map: Map with time horizon control
        """
        if not predictions:
            if self.verbose:
                print("No predictions provided")
            return m
        
        # Filter predictions by target ID
        target_preds = {}
        
        for key, pred in predictions.items():
            if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                # If methods specified, filter by method
                if methods is not None and 'method' in pred:
                    if pred['method'] in methods:
                        horizon = pred['minutes_ahead']
                        target_preds[horizon] = pred
                else:
                    horizon = pred['minutes_ahead']
                    target_preds[horizon] = pred
        
        if not target_preds:
            if self.verbose:
                print(f"No predictions found for target {target_id}")
            return m
        
        # Sort time horizons
        horizons = sorted(target_preds.keys())
        
        # Create a layer for each time horizon
        for horizon in horizons:
            pred = target_preds[horizon]
            
            # Get method if available
            method = pred.get('method', None)
            
            # Add prediction to map
            self.add_prediction_heatmap(m, pred, target_id, horizon, method)
        
        return m
    
    def add_methods_control(self, m, predictions, target_id, time_horizon=None):
        """
        Add a control to switch between prediction methods.
        
        Args:
            m (folium.Map): Map to add control to
            predictions (dict): Dictionary of prediction results
            target_id: Target ID to show predictions for
            time_horizon (int): Time horizon to display (optional)
            
        Returns:
            folium.Map: Map with methods control
        """
        if not predictions:
            if self.verbose:
                print("No predictions provided")
            return m
        
        # Filter predictions by target ID
        target_preds = {}
        
        for key, pred in predictions.items():
            if isinstance(pred, dict) and 'target_id' in pred and pred['target_id'] == target_id:
                # If time horizon specified, filter by time horizon
                if time_horizon is not None and 'minutes_ahead' in pred:
                    if pred['minutes_ahead'] == time_horizon:
                        method = pred.get('method', 'unknown')
                        target_preds[method] = pred
                else:
                    method = pred.get('method', 'unknown')
                    target_preds[method] = pred
        
        if not target_preds:
            if self.verbose:
                print(f"No predictions found for target {target_id}")
            return m
        
        # Create a layer for each method
        for method, pred in target_preds.items():
            # Get time horizon if available
            horizon = pred.get('minutes_ahead', None)
            
            # Add prediction to map
            self.add_prediction_heatmap(m, pred, target_id, horizon, method)
        
        return m
    
    def add_dual_map_comparison(self, m_width, m_height, predictions1, predictions2, 
                               target_id, time_horizon=None, method=None, 
                               title1="Prediction A", title2="Prediction B"):
        """
        Create a dual map for side-by-side comparison.
        
        Args:
            m_width (str): Width of each map
            m_height (str): Height of each map
            predictions1 (dict): First set of prediction results
            predictions2 (dict): Second set of prediction results
            target_id: Target ID to show predictions for
            time_horizon (int): Time horizon to display (optional)
            method (str): Prediction method to display (optional)
            title1 (str): Title for first map
            title2 (str): Title for second map
            
        Returns:
            folium.plugins.DualMap: Dual map with side-by-side comparison
        """
        # Find center point
        center = None
        if not self.targets_df.empty:
            target_data = self.targets_df[self.targets_df['target_id'] == target_id]
            if not target_data.empty:
                center = [
                    target_data['latitude'].mean(),
                    target_data['longitude'].mean()
                ]
        
        if center is None and not self.targets_df.empty:
            center = [
                self.targets_df['latitude'].mean(),
                self.targets_df['longitude'].mean()
            ]
        
        if center is None:
            center = [0, 0]
        
        # Create dual map
        dual_map = DualMap(
            location=center,
            zoom_start=self.config['zoom_start'],
            tiles=self.config['map_style'],
            width=m_width,
            height=m_height
        )
        
        # Extract the two map objects from the dual map
        m1 = dual_map.m1
        m2 = dual_map.m2
        
        # Add background layers to both maps
        for tile_key, tile_name in self.config['map_tiles'].items():
            if tile_name != self.config['map_style']:  # Skip the default tile we already added
                folium.TileLayer(tile_name).add_to(m1)
                folium.TileLayer(tile_name).add_to(m2)
        
        # Add layer controls
        folium.LayerControl().add_to(m1)
        folium.LayerControl().add_to(m2)
        
        # Add measure controls if configured
        if self.config['enable_measure']:
            MeasureControl().add_to(m1)
            MeasureControl().add_to(m2)
        
        # Add target trails to both maps
        self.add_target_trails(m1, [target_id], show_markers=True, layer_name="Target Track")
        self.add_target_trails(m2, [target_id], show_markers=True, layer_name="Target Track")
        
        # Add blue forces to both maps
        self.add_blue_forces(m1)
        self.add_blue_forces(m2)
        
        # Filter predictions
        pred1 = self._filter_prediction(predictions1, target_id, time_horizon, method)
        pred2 = self._filter_prediction(predictions2, target_id, time_horizon, method)
        
        # Add predictions to maps
        if pred1 is not None:
            self.add_prediction_heatmap(m1, pred1, target_id, time_horizon, method, layer_name=title1)
        
        if pred2 is not None:
            self.add_prediction_heatmap(m2, pred2, target_id, time_horizon, method, layer_name=title2)
        
        # Add titles in JavaScript
        title_js = f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var title1 = document.createElement('div');
            title1.className = 'map-title';
            title1.style.position = 'absolute';
            title1.style.zIndex = '999';
            title1.style.fontSize = '16px';
            title1.style.fontWeight = 'bold';
            title1.style.backgroundColor = 'white';
            title1.style.padding = '5px';
            title1.style.borderRadius = '5px';
            title1.style.top = '10px';
            title1.style.left = '50px';
            title1.style.boxShadow = '0 1px 5px rgba(0,0,0,0.4)';
            title1.innerHTML = '{title1}';
            document.querySelector('.leaflet-sbs-leftmap').appendChild(title1);
            
            var title2 = document.createElement('div');
            title2.className = 'map-title';
            title2.style.position = 'absolute';
            title2.style.zIndex = '999';
            title2.style.fontSize = '16px';
            title2.style.fontWeight = 'bold';
            title2.style.backgroundColor = 'white';
            title2.style.padding = '5px';
            title2.style.borderRadius = '5px';
            title2.style.top = '10px';
            title2.style.left = '50px';
            title2.style.boxShadow = '0 1px 5px rgba(0,0,0,0.4)';
            title2.innerHTML = '{title2}';
            document.querySelector('.leaflet-sbs-rightmap').appendChild(title2);
        }});
        </script>
        """
        dual_map.get_root().html.add_child(folium.Element(title_js))
        
        return dual_map
    
    def create_dashboard_map(self, target_ids=None, time_horizons=None, methods=None,
                           width="100%", height="800px", include_controls=True,
                           include_terrain=True, include_land_use=True):
        """
        Create a comprehensive dashboard map with multiple layers and controls.
        
        Args:
            target_ids (list): List of target IDs to include (optional)
            time_horizons (list): List of time horizons to include (optional)
            methods (list): List of methods to include (optional)
            width (str): Width of the map
            height (str): Height of the map
            include_controls (bool): Whether to include additional controls
            include_terrain (bool): Whether to include terrain overlay
            include_land_use (bool): Whether to include land use overlay
            
        Returns:
            folium.Map: Dashboard map
        """
        # Create base map
        m = self.create_base_map(width=width, height=height)
        
        # Add terrain overlay if requested
        if include_terrain and not self.terrain_grid_df.empty:
            self.add_terrain_overlay(m, feature_type='elevation')
            self.add_terrain_overlay(m, layer_name="Terrain Cost", feature_type='cost')
            self.add_terrain_overlay(m, layer_name="Concealment", feature_type='concealment')
        
        # Add land use if requested
        if include_land_use and not self.terrain_grid_df.empty and 'land_use_type' in self.terrain_grid_df.columns:
            self.add_land_use(m)
        
        # Add targets
        # Determine target IDs if not provided
        if target_ids is None:
            if 'is_blue' in self.targets_df.columns:
                targets_subset = self.targets_df[self.targets_df['is_blue'] == 0].copy()
            else:
                targets_subset = self.targets_df.copy()
            
            target_ids = targets_subset['target_id'].unique()
        
        # Add target trails
        self.add_target_trails(m, target_ids, show_markers=True, animate=True)
        
        # Add blue forces
        self.add_blue_forces(m)
        
        # Add predictions if available
        if self.predictions:
            # Determine time horizons if not provided
            if time_horizons is None:
                available_horizons = set()
                for key, pred in self.predictions.items():
                    if isinstance(pred, dict) and 'minutes_ahead' in pred:
                        available_horizons.add(pred['minutes_ahead'])
                
                time_horizons = sorted(available_horizons)
            
            # Determine methods if not provided
            if methods is None:
                available_methods = set()
                for key, pred in self.predictions.items():
                    if isinstance(pred, dict) and 'method' in pred:
                        available_methods.add(pred['method'])
                
                methods = sorted(available_methods)
            
            # Add predictions for each target, time horizon, and method
            for target_id in target_ids:
                for horizon in time_horizons:
                    for method in methods:
                        # Find matching prediction
                        pred = None
                        for key, p in self.predictions.items():
                            if (isinstance(p, dict) and 'target_id' in p and p['target_id'] == target_id and
                                'minutes_ahead' in p and p['minutes_ahead'] == horizon and
                                'method' in p and p['method'] == method):
                                pred = p
                                break
                        
                        if pred is not None:
                            self.add_prediction_heatmap(m, pred, target_id, horizon, method)
        
        # Add additional controls if requested
        if include_controls:
            # Add control for current time display
            current_time_html = """
            <div id="current-time" style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                     background-color: white; padding: 5px 10px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                     font-size: 14px; z-index: 1000;">
                Current time: Loading...
            </div>
            
            <script>
            function updateCurrentTime() {
                var now = new Date();
                var timeString = now.toLocaleString();
                document.getElementById('current-time').innerHTML = 'Current time: ' + timeString;
            }
            
            // Update time immediately
            updateCurrentTime();
            
            // Update time every second
            setInterval(updateCurrentTime, 1000);
            </script>
            """
            m.get_root().html.add_child(folium.Element(current_time_html))
            
            # Add info panel
            info_html = """
            <div id="info-panel" style="position: fixed; top: 10px; right: 10px; width: 300px; max-height: 80%; 
                     background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                     font-size: 14px; z-index: 1000; overflow-y: auto; display: none;">
                <h3 style="margin-top: 0;">Predictive Tracking Dashboard</h3>
                <p>This interactive dashboard provides visualizations of predicted target movements.</p>
                <p>Use the layer control to toggle different views:</p>
                <ul>
                    <li><strong>Target Tracks:</strong> Historical target movements</li>
                    <li><strong>Blue Forces:</strong> Friendly force positions</li>
                    <li><strong>Predictions:</strong> Probability heatmaps for future locations</li>
                    <li><strong>Terrain:</strong> Elevation, cost, and concealment overlays</li>
                    <li><strong>Land Use:</strong> Urban, forest, road, and water areas</li>
                </ul>
                <p>Use the measure tool to calculate distances and areas.</p>
                <p>Draw tools allow creating custom annotations.</p>
                <button onclick="document.getElementById('info-panel').style.display='none'" 
                        style="padding: 5px 10px; background-color: #f8f9fa; border: 1px solid #dee2e6; 
                        border-radius: 3px; cursor: pointer;">Close</button>
            </div>
            
            <button id="info-button" onclick="document.getElementById('info-panel').style.display='block'" 
                    style="position: fixed; top: 10px; right: 10px; width: 40px; height: 40px; 
                    background-color: white; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                    border: none; font-size: 20px; cursor: pointer; z-index: 999;">ℹ️</button>
            """
            m.get_root().html.add_child(folium.Element(info_html))
        
        return m
    
    def create_3d_terrain_map(self, center=None, zoom_start=None, target_id=None, prediction=None,
                             width=800, height=600, exaggeration=1.5):
        """
        Create a 3D terrain map using Plotly.
        
        Args:
            center (tuple): Center coordinates (lat, lon) (optional)
            zoom_start (float): Initial zoom level
            target_id: Target ID to show (optional)
            prediction (dict): Prediction data to overlay (optional)
            width (int): Width of the map in pixels
            height (int): Height of the map in pixels
            exaggeration (float): Vertical exaggeration factor
            
        Returns:
            plotly.graph_objects.Figure: 3D terrain map
        """
        if self.terrain_grid_df.empty or 'elevation' not in self.terrain_grid_df.columns:
            if self.verbose:
                print("No terrain data available for 3D map")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Reshape terrain data to grid
        try:
            # Get grid size (assuming square grid)
            grid_size = int(np.sqrt(len(self.terrain_grid_df)))
            
            # Reshape terrain data
            lats = self.terrain_grid_df['latitude'].values.reshape(grid_size, grid_size)
            lons = self.terrain_grid_df['longitude'].values.reshape(grid_size, grid_size)
            elevs = self.terrain_grid_df['elevation'].values.reshape(grid_size, grid_size)
            
            # Apply vertical exaggeration
            elevs = elevs * exaggeration
            
            # Determine color scale based on land use if available
            if 'land_use_type' in self.terrain_grid_df.columns:
                # Get land use types
                land_use = self.terrain_grid_df['land_use_type'].values.reshape(grid_size, grid_size)
                
                # Create a categorical color mapping
                land_use_map = {
                    'urban': '#A9A9A9',     # Dark gray
                    'road': '#000000',      # Black
                    'forest': '#228B22',    # Forest green
                    'open': '#90EE90',      # Light green
                    'water': '#1E90FF',     # Dodger blue
                    'wetland': '#7D9EC0',   # Light slate
                    'restricted': '#FF8C00'  # Dark orange
                }
                
                # Map land use types to colors
                color_array = np.zeros(land_use.shape, dtype=object)
                for i in range(grid_size):
                    for j in range(grid_size):
                        color_array[i, j] = land_use_map.get(land_use[i, j], '#CCCCCC')
                
                # Create 3D surface plot with land use colors
                fig.add_trace(go.Surface(
                    z=elevs,
                    x=lons,
                    y=lats,
                    colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'brown']],
                    surfacecolor=elevs,
                    showscale=True,
                    opacity=0.8,
                    contours={
                        'z': {
                            'show': True,
                            'start': elevs.min(),
                            'end': elevs.max(),
                            'size': (elevs.max() - elevs.min()) / 10,
                            'color': 'black'
                        }
                    }
                ))
                
                # Add land use legend
                for land_use_type, color in land_use_map.items():
                    if land_use_type in land_use:
                        fig.add_trace(go.Scatter3d(
                            x=[None],
                            y=[None],
                            z=[None],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=land_use_type.capitalize(),
                            showlegend=True
                        ))
            else:
                # Use elevation for color if no land use data
                fig.add_trace(go.Surface(
                    z=elevs,
                    x=lons,
                    y=lats,
                    colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'brown']],
                    showscale=True,
                    opacity=0.8,
                    contours={
                        'z': {
                            'show': True,
                            'start': elevs.min(),
                            'end': elevs.max(),
                            'size': (elevs.max() - elevs.min()) / 10,
                            'color': 'black'
                        }
                    }
                ))
            
            # Add target track if requested
            if target_id is not None and not self.targets_df.empty:
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
                    
                    # Add line for target track
                    fig.add_trace(go.Scatter3d(
                        x=target_data['longitude'],
                        y=target_data['latitude'],
                        z=target_data['elevation'] * exaggeration if 'elevation' in target_data.columns
                            else np.interp(target_data.index, 
                                          np.arange(len(target_data)), 
                                          np.linspace(elevs.min() + 10, elevs.max() + 10, len(target_data))),
                        mode='lines+markers',
                        line=dict(color=color, width=5),
                        marker=dict(size=5, color=color),
                        name=f"Target {target_id} Track"
                    ))
            
            # Add prediction overlay if provided
            if prediction is not None and 'density' in prediction:
                # Extract prediction data
                lat_grid_pred = prediction['lat_grid']
                lon_grid_pred = prediction['lon_grid']
                density = prediction['density']
                
                # Normalize density for visualization
                norm_density = density / density.max() if density.max() > 0 else density
                
                # Create a meshgrid for the prediction
                lon_mesh, lat_mesh = np.meshgrid(lon_grid_pred, lat_grid_pred)
                
                # Interpolate elevation at prediction points
                from scipy.interpolate import griddata
                elev_interp = griddata(
                    (lons.flatten(), lats.flatten()),
                    elevs.flatten(),
                    (lon_mesh, lat_mesh),
                    method='linear',
                    fill_value=elevs.mean()
                )
                
                # Create a 3D scatter plot with prediction density
                points = []
                colors = []
                sizes = []
                
                for i in range(len(lat_grid_pred)):
                    for j in range(len(lon_grid_pred)):
                        if norm_density[i, j] > 0.1:  # Filter out low probability points
                            points.append((lon_mesh[i, j], lat_mesh[i, j], elev_interp[i, j] + 20))
                            
                            # Determine color based on density (blue to red)
                            r = min(1.0, norm_density[i, j] * 2)
                            g = max(0, 1 - norm_density[i, j])
                            b = max(0, 1 - norm_density[i, j] * 2)
                            colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
                            
                            # Determine size based on density
                            sizes.append(norm_density[i, j] * 15 + 5)
                
                if points:
                    x, y, z = zip(*points)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=colors,
                            opacity=0.7
                        ),
                        name="Prediction Density"
                    ))
        except Exception as e:
            if self.verbose:
                print(f"Error creating 3D terrain map: {str(e)}")
            return None
        
        # Set layout
        fig.update_layout(
            width=width,
            height=height,
            title=f"3D Terrain Map{f' - Target {target_id}' if target_id else ''}",
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def save_map(self, m, filename=None, format='html'):
        """
        Save the map to a file.
        
        Args:
            m: Map to save (folium.Map or plotly.graph_objects.Figure)
            filename (str): Output filename (optional)
            format (str): Output format ('html', 'png', 'pdf', 'json')
            
        Returns:
            str: Output filename
        """
        # Create default filename if not provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.config['output_folder'], f"map_{timestamp}.{format}")
        
        # Ensure output folder exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save based on type
        if isinstance(m, folium.Map) or isinstance(m, DualMap):
            if format == 'html':
                m.save(filename)
            elif format in ['png', 'jpg', 'jpeg', 'pdf']:
                # For image formats, we need to capture a screenshot
                # This requires a web browser, but we'll save as HTML first
                html_file = filename.replace(f".{format}", ".html")
                m.save(html_file)
                
                # Note: rendering to image would require a browser automation like Selenium
                if self.verbose:
                    print(f"Saved map as HTML to {html_file}")
                    print(f"Note: Direct rendering to {format} requires browser automation")
                
                return html_file
            else:
                if self.verbose:
                    print(f"Unsupported format for folium map: {format}, saving as HTML")
                
                # Default to HTML
                if not filename.endswith('.html'):
                    filename = filename + '.html'
                
                m.save(filename)
        elif isinstance(m, go.Figure):
            # Plotly figure
            if format == 'html':
                m.write_html(filename)
            elif format == 'png':
                m.write_image(filename)
            elif format == 'pdf':
                m.write_image(filename)
            elif format == 'json':
                import json
                with open(filename, 'w') as f:
                    json.dump(m.to_dict(), f)
            else:
                if self.verbose:
                    print(f"Unsupported format for plotly figure: {format}, saving as HTML")
                
                # Default to HTML
                if not filename.endswith('.html'):
                    filename = filename + '.html'
                
                m.write_html(filename)
        else:
            if self.verbose:
                print(f"Unsupported map type: {type(m)}")
            return None
        
        if self.verbose:
            print(f"Saved map to {filename}")
        
        return filename
    
    def create_map_comparison_html(self, maps, titles=None, layout="grid", width="100%", height="800px"):
        """
        Create an HTML page with multiple maps for comparison.
        
        Args:
            maps (list): List of maps (folium.Map or HTML strings)
            titles (list): List of titles for each map (optional)
            layout (str): Layout type ('grid', 'tabs')
            width (str): Width of the map container
            height (str): Height of the map container
            
        Returns:
            str: HTML string with embedded maps
        """
        if not maps:
            if self.verbose:
                print("No maps provided")
            return None
        
        # Create default titles if not provided
        if titles is None:
            titles = [f"Map {i+1}" for i in range(len(maps))]
        
        # Ensure titles match the number of maps
        if len(titles) != len(maps):
            titles = titles[:len(maps)]
            while len(titles) < len(maps):
                titles.append(f"Map {len(titles)+1}")
        
        # Create HTML header
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Map Comparison</title>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }
                .container {
                    width: 100%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    text-align: center;
                    margin-bottom: 20px;
                }
        """
        
        # Add layout-specific styles
        if layout == "grid":
            # Determine grid layout based on number of maps
            if len(maps) == 1:
                cols = 1
            elif len(maps) == 2:
                cols = 2
            elif len(maps) in [3, 4]:
                cols = 2
            else:
                cols = 3
            
            html += f"""
                .map-grid {{
                    display: grid;
                    grid-template-columns: repeat({cols}, 1fr);
                    gap: 20px;
                }}
                .map-item {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                .map-title {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                    text-align: center;
                    font-weight: bold;
                }}
                .map-content {{
                    height: {height};
                }}
            """
        elif layout == "tabs":
            html += """
                .tab {
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                }
                .tab button {
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                    font-size: 16px;
                }
                .tab button:hover {
                    background-color: #ddd;
                }
                .tab button.active {
                    background-color: #ccc;
                }
                .tabcontent {
                    display: none;
                    border: 1px solid #ccc;
                    border-top: none;
                }
                .tabcontent.active {
                    display: block;
                }
                .map-content {
                    height: """ + height + """;
                }
            """
        
        html += """
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Map Comparison</h1>
        """
        
        # Add maps based on layout
        if layout == "grid":
            html += '<div class="map-grid">'
            
            for i, m in enumerate(maps):
                html += f"""
                <div class="map-item">
                    <div class="map-title">{titles[i]}</div>
                    <div class="map-content" id="map-{i+1}">
                """
                
                # Add map content
                if isinstance(m, folium.Map):
                    # Save map to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                        m.save(tmp.name)
                        with open(tmp.name, 'r') as f:
                            map_html = f.read()
                        
                        # Extract the map div and script
                        import re
                        map_div = re.search(r'<div class="folium-map".*?</div>', map_html, re.DOTALL)
                        map_script = re.search(r'<script>.*?</script>', map_html, re.DOTALL)
                        
                        if map_div and map_script:
                            html += map_div.group(0) + map_script.group(0)
                        else:
                            html += "<p>Error: Unable to extract map content</p>"
                        
                        # Clean up temporary file
                        os.unlink(tmp.name)
                elif isinstance(m, str):
                    # Assume it's an HTML string
                    html += m
                else:
                    html += f"<p>Unsupported map type: {type(m)}</p>"
                
                html += """
                    </div>
                </div>
                """
            
            html += '</div>'
        elif layout == "tabs":
            # Add tab buttons
            html += '<div class="tab">'
            for i, title in enumerate(titles):
                active = ' class="active"' if i == 0 else ''
                html += f'<button{active} onclick="openTab(event, \'map-{i+1}\')">{title}</button>'
            html += '</div>'
            
            # Add tab content
            for i, m in enumerate(maps):
                active = ' active' if i == 0 else ''
                html += f'<div id="map-{i+1}" class="tabcontent{active}">'
                html += f'<div class="map-content">'
                
                # Add map content
                if isinstance(m, folium.Map):
                    # Save map to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                        m.save(tmp.name)
                        with open(tmp.name, 'r') as f:
                            map_html = f.read()
                        
                        # Extract the map div and script
                        import re
                        map_div = re.search(r'<div class="folium-map".*?</div>', map_html, re.DOTALL)
                        map_script = re.search(r'<script>.*?</script>', map_html, re.DOTALL)
                        
                        if map_div and map_script:
                            html += map_div.group(0) + map_script.group(0)
                        else:
                            html += "<p>Error: Unable to extract map content</p>"
                        
                        # Clean up temporary file
                        os.unlink(tmp.name)
                elif isinstance(m, str):
                    # Assume it's an HTML string
                    html += m
                else:
                    html += f"<p>Unsupported map type: {type(m)}</p>"
                
                html += '</div></div>'
            
            # Add tab JavaScript
            html += """
            <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
                
                // Trigger resize to fix map rendering issues
                window.dispatchEvent(new Event('resize'));
            }
            </script>
            """
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def convert_to_geojson(self, feature_type='targets', target_ids=None, include_predictions=False,
                          time_horizon=None, method=None, filename=None):
        """
        Convert map data to GeoJSON format.
        
        Args:
            feature_type (str): Type of features to include ('targets', 'predictions', 'terrain', 'all')
            target_ids (list): List of target IDs to include (optional)
            include_predictions (bool): Whether to include prediction data
            time_horizon (int): Time horizon for predictions (optional)
            method (str): Prediction method (optional)
            filename (str): Output filename (optional)
            
        Returns:
            dict: GeoJSON data
        """
        # Create empty GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Add targets if requested
        if feature_type in ['targets', 'all'] and not self.targets_df.empty:
            # Filter targets if target_ids provided
            if target_ids is not None:
                targets_subset = self.targets_df[self.targets_df['target_id'].isin(target_ids)].copy()
            else:
                targets_subset = self.targets_df.copy()
            
            # Filter out blue forces from targets
            if 'is_blue' in targets_subset.columns:
                targets_subset = targets_subset[targets_subset['is_blue'] == 0].copy()
            
            # Get unique target IDs
            unique_targets = sorted(targets_subset['target_id'].unique())
            
            for target_id in unique_targets:
                target_data = targets_subset[targets_subset['target_id'] == target_id].copy()
                
                # Skip if no data
                if target_data.empty:
                    continue
                
                # Sort by timestamp
                target_data = target_data.sort_values('timestamp')
                
                # Get target class
                if 'target_class' in target_data.columns:
                    target_class = target_data['target_class'].iloc[0]
                else:
                    target_class = 'unknown'
                
                # Create line feature for target track
                coords = [[row['longitude'], row['latitude']] for _, row in target_data.iterrows()]
                
                if len(coords) >= 2:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        },
                        "properties": {
                            "target_id": target_id,
                            "target_class": target_class,
                            "type": "target_track",
                            "start_time": target_data['timestamp'].min().isoformat(),
                            "end_time": target_data['timestamp'].max().isoformat()
                        }
                    }
                    
                    geojson["features"].append(feature)
                
                # Add point features for each observation
                for _, row in target_data.iterrows():
                    point_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row['longitude'], row['latitude']]
                        },
                        "properties": {
                            "target_id": target_id,
                            "target_class": target_class,
                            "type": "target_observation",
                            "timestamp": row['timestamp'].isoformat()
                        }
                    }
                    
                    # Add additional properties if available
                    if 'speed' in row:
                        point_feature["properties"]["speed"] = float(row['speed'])
                    
                    if 'heading' in row:
                        point_feature["properties"]["heading"] = float(row['heading'])
                    
                    geojson["features"].append(point_feature)
        
        # Add blue forces if requested
        if feature_type in ['targets', 'all'] and not self.blue_forces_df.empty:
            # Get most recent position for each blue force
            if 'timestamp' in self.blue_forces_df.columns:
                blue_latest = self.blue_forces_df.sort_values('timestamp').groupby('blue_id').last().reset_index()
            else:
                blue_latest = self.blue_forces_df.copy()
            
            # Add point features for each blue force
            for _, row in blue_latest.iterrows():
                # Get blue force class
                blue_class = row['blue_class'] if 'blue_class' in row else 'unknown'
                
                point_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row['longitude'], row['latitude']]
                    },
                    "properties": {
                        "blue_id": row['blue_id'],
                        "blue_class": blue_class,
                        "type": "blue_force"
                    }
                }
                
                # Add timestamp if available
                if 'timestamp' in row:
                    point_feature["properties"]["timestamp"] = row['timestamp'].isoformat()
                
                # Add additional properties if available
                if 'heading' in row:
                    point_feature["properties"]["heading"] = float(row['heading'])
                
                geojson["features"].append(point_feature)
        
        # Add predictions if requested
        if (feature_type in ['predictions', 'all'] or include_predictions) and self.predictions:
            # Filter targets if target_ids provided
            target_ids_to_use = target_ids if target_ids is not None else []
            
            # If no target_ids provided, use all targets
            if not target_ids_to_use and not self.targets_df.empty:
                if 'is_blue' in self.targets_df.columns:
                    targets_subset = self.targets_df[self.targets_df['is_blue'] == 0].copy()
                else:
                    targets_subset = self.targets_df.copy()
                
                target_ids_to_use = sorted(targets_subset['target_id'].unique())
            
            # Add predictions for each target
            for target_id in target_ids_to_use:
                # Find matching prediction
                pred = None
                for key, p in self.predictions.items():
                    if not isinstance(p, dict) or 'target_id' not in p:
                        continue
                    
                    if p['target_id'] != target_id:
                        continue
                    
                    if time_horizon is not None and 'minutes_ahead' in p:
                        if p['minutes_ahead'] != time_horizon:
                            continue
                    
                    if method is not None and 'method' in p:
                        if p['method'] != method:
                            continue
                    
                    pred = p
                    break
                
                if pred is not None and 'density' in pred:
                    # Extract prediction data
                    lat_grid = pred['lat_grid']
                    lon_grid = pred['lon_grid']
                    density = pred['density']
                    
                    # Get metadata
                    pred_time_horizon = pred.get('minutes_ahead', None)
                    pred_method = pred.get('method', None)
                    
                    # Find maximum density point
                    max_idx = np.argmax(density)
                    max_i, max_j = np.unravel_index(max_idx, density.shape)
                    max_lat = lat_grid[max_i]
                    max_lon = lon_grid[max_j]
                    
                    # Add most likely position point
                    point_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [max_lon, max_lat]
                        },
                        "properties": {
                            "target_id": target_id,
                            "type": "prediction_max",
                            "time_horizon": pred_time_horizon,
                            "method": pred_method,
                            "probability": float(density[max_i, max_j])
                        }
                    }
                    
                    geojson["features"].append(point_feature)
                    
                    # Add confidence regions if available
                    if 'confidence_regions' in pred:
                        for level, region in pred['confidence_regions'].items():
                            if 'points' in region and len(region['points']) >= 3:
                                # Create polygon
                                try:
                                    from scipy.spatial import ConvexHull
                                    hull = ConvexHull(region['points'])
                                    hull_points = [region['points'][i] for i in hull.vertices]
                                    
                                    # Add polygon to GeoJSON
                                    polygon_feature = {
                                        "type": "Feature",
                                        "geometry": {
                                            "type": "Polygon",
                                            "coordinates": [[[p[1], p[0]] for p in hull_points]]
                                        },
                                        "properties": {
                                            "target_id": target_id,
                                            "type": "confidence_region",
                                            "confidence_level": float(level),
                                            "time_horizon": pred_time_horizon,
                                            "method": pred_method
                                        }
                                    }
                                    
                                    # Add area if available
                                    if 'area_km2' in region:
                                        polygon_feature["properties"]["area_km2"] = region['area_km2']
                                    
                                    geojson["features"].append(polygon_feature)
                                except:
                                    # Skip if convex hull fails
                                    pass
        
        # Add terrain data if requested
        if feature_type in ['terrain', 'all'] and not self.terrain_grid_df.empty:
            if 'land_use_type' in self.terrain_grid_df.columns:
                # Group by land use type
                for land_use_type in self.terrain_grid_df['land_use_type'].unique():
                    # Get points for this land use type
                    type_points = self.terrain_grid_df[self.terrain_grid_df['land_use_type'] == land_use_type]
                    
                    # Try to create a polygon
                    try:
                        # Create points
                        points = [Point(row['longitude'], row['latitude']) for _, row in type_points.iterrows()]
                        
                        # Buffer points to create areas
                        areas = [p.buffer(0.001) for p in points]
                        
                        # Combine overlapping areas
                        from shapely.ops import unary_union # type: ignore
                        combined_areas = unary_union(areas)
                        
                        # Add to GeoJSON
                        if hasattr(combined_areas, 'geoms'):
                            # Multiple polygons
                            for poly in combined_areas.geoms:
                                # Convert to list of coordinates
                                coords = [[[y, x] for x, y in zip(*poly.exterior.xy)]]
                                
                                # Create polygon feature
                                polygon_feature = {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": coords
                                    },
                                    "properties": {
                                        "type": "land_use",
                                        "land_use_type": land_use_type
                                    }
                                }
                                
                                geojson["features"].append(polygon_feature)
                        else:
                            # Single polygon
                            coords = [[[y, x] for x, y in zip(*combined_areas.exterior.xy)]]
                            
                            # Create polygon feature
                            polygon_feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": coords
                                },
                                "properties": {
                                    "type": "land_use",
                                    "land_use_type": land_use_type
                                }
                            }
                            
                            geojson["features"].append(polygon_feature)
                    except:
                        # If combining fails, add individual points
                        for _, row in type_points.iterrows():
                            point_feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [row['longitude'], row['latitude']]
                                },
                                "properties": {
                                    "type": "terrain_point",
                                    "land_use_type": land_use_type
                                }
                            }
                            
                            # Add additional properties if available
                            if 'elevation' in row:
                                point_feature["properties"]["elevation"] = float(row['elevation'])
                            
                            if 'total_cost' in row:
                                point_feature["properties"]["terrain_cost"] = float(row['total_cost'])
                            
                            if 'concealment' in row:
                                point_feature["properties"]["concealment"] = float(row['concealment'])
                            
                            geojson["features"].append(point_feature)
            else:
                # Add terrain points
                for _, row in self.terrain_grid_df.iterrows():
                    point_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row['longitude'], row['latitude']]
                        },
                        "properties": {
                            "type": "terrain_point"
                        }
                    }
                    
                    # Add additional properties if available
                    if 'elevation' in row:
                        point_feature["properties"]["elevation"] = float(row['elevation'])
                    
                    if 'total_cost' in row:
                        point_feature["properties"]["terrain_cost"] = float(row['total_cost'])
                    
                    if 'concealment' in row:
                        point_feature["properties"]["concealment"] = float(row['concealment'])
                    
                    geojson["features"].append(point_feature)
        
        # Save to file if filename provided
        if filename is not None:
            with open(filename, 'w') as f:
                import json
                json.dump(geojson, f, indent=2)
            
            if self.verbose:
                print(f"Saved GeoJSON to {filename}")
        
        return geojson
    
    def _create_heading_icon(self, color, heading):
        """
        Create an HTML string for an icon showing heading.
        
        Args:
            color (str): Color for the icon
            heading (float): Heading angle in degrees
            
        Returns:
            str: HTML for the icon
        """
        # Calculate the arrowhead coordinates
        cx, cy = 15, 15  # Center of the circle
        r = 12  # Radius
        
        # Convert heading from degrees to radians (0° is North, 90° is East)
        angle_rad = math.radians(90 - heading)
        
        # Calculate arrow endpoint
        arrow_x = cx + r * math.cos(angle_rad)
        arrow_y = cy - r * math.sin(angle_rad)
        
        # Calculate arrowhead points
        arrow_size = 5
        arrow_angle1 = angle_rad + math.radians(150)
        arrow_angle2 = angle_rad - math.radians(150)
        
        arrow_x1 = arrow_x + arrow_size * math.cos(arrow_angle1)
        arrow_y1 = arrow_y - arrow_size * math.sin(arrow_angle1)
        arrow_x2 = arrow_x + arrow_size * math.cos(arrow_angle2)
        arrow_y2 = arrow_y - arrow_size * math.sin(arrow_angle2)
        
        # Create SVG for the icon
        svg = f"""
        <svg width="30" height="30" viewBox="0 0 30 30" xmlns="http://www.w3.org/2000/svg">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="{color}" stroke-width="1" />
            <line x1="{cx}" y1="{cy}" x2="{arrow_x}" y2="{arrow_y}" stroke="{color}" stroke-width="2" />
            <line x1="{arrow_x}" y1="{arrow_y}" x2="{arrow_x1}" y2="{arrow_y1}" stroke="{color}" stroke-width="2" />
            <line x1="{arrow_x}" y1="{arrow_y}" x2="{arrow_x2}" y2="{arrow_y2}" stroke="{color}" stroke-width="2" />
        </svg>
        """
        
        return svg
    
    def _blend_colors(self, color1, color2, ratio):
        """
        Blend two colors based on the given ratio.
        
        Args:
            color1 (str): First color (hex or name)
            color2 (str): Second color (hex or name)
            ratio (float): Blend ratio (0.0 to 1.0), 0 = full color1, 1 = full color2
            
        Returns:
            str: Blended color in hex format
        """
        # Convert color names to hex
        color1 = self._name_to_hex(color1)
        color2 = self._name_to_hex(color2)
        
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        # Blend colors
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _name_to_hex(self, color):
        """
        Convert color name to hex code.
        
        Args:
            color (str): Color name or hex code
            
        Returns:
            str: Hex color code
        """
        # Return if already hex
        if color.startswith('#'):
            return color
        
        # Common color names to hex mapping
        color_map = {
            'red': '#FF0000',
            'green': '#008000',
            'blue': '#0000FF',
            'yellow': '#FFFF00',
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            'black': '#000000',
            'white': '#FFFFFF',
            'gray': '#808080',
            'orange': '#FFA500',
            'purple': '#800080',
            'brown': '#A52A2A',
            'pink': '#FFC0CB'
        }
        
        # Return hex if found in map
        if color.lower() in color_map:
            return color_map[color.lower()]
        
        # Default to black if not found
        return '#000000'
    
    def _color_to_name(self, color):
        """
        Convert color to name for Folium icons.
        
        Args:
            color (str): Color hex or name
            
        Returns:
            str: Color name for Folium icon
        """
        # Map of common color hex to Folium icon names
        folium_colors = {
            '#FF0000': 'red',
            '#008000': 'green',
            '#0000FF': 'blue',
            '#FFFF00': 'yellow',
            '#FFA500': 'orange',
            '#800080': 'purple',
            '#A52A2A': 'brown',
            '#000000': 'black',
            '#FFFFFF': 'white',
            '#808080': 'gray',
            '#FFC0CB': 'pink',
            'red': 'red',
            'green': 'green',
            'blue': 'blue',
            'yellow': 'yellow',
            'orange': 'orange',
            'purple': 'purple',
            'brown': 'brown',
            'black': 'black',
            'white': 'white',
            'gray': 'gray',
            'pink': 'pink'
        }
        
        # Convert hex to name
        if color in folium_colors:
            return folium_colors[color]
        
        # Default to 'red' if color not found
        return 'red'
    
    def _filter_prediction(self, predictions, target_id, time_horizon=None, method=None):
        """
        Filter predictions by target ID, time horizon, and method.
        
        Args:
            predictions (dict): Dictionary of prediction results
            target_id: Target ID to filter for
            time_horizon (int): Time horizon in minutes (optional)
            method (str): Prediction method (optional)
            
        Returns:
            dict: Filtered prediction data
        """
        if not predictions:
            return None
        
        for key, pred in predictions.items():
            if not isinstance(pred, dict) or 'target_id' not in pred:
                continue
            
            if pred['target_id'] != target_id:
                continue
            
            if time_horizon is not None and 'minutes_ahead' in pred:
                if pred['minutes_ahead'] != time_horizon:
                    continue
            
            if method is not None and 'method' in pred:
                if pred['method'] != method:
                    continue
            
            return pred
        
        return None