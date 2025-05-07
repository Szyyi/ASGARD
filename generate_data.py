import pandas as pd 
import numpy as np
import datetime
import os
from tqdm import tqdm # type: ignore

def generate_large_target_dataset(num_targets=100, days=14, sample_interval_minutes=5, output_file="data/target_data.csv"):
    """
    Generate a large synthetic dataset of target movements.
    
    Parameters:
    - num_targets: Number of unique targets to generate
    - days: Number of days of data to generate
    - sample_interval_minutes: Time between observations in minutes
    - output_file: Path to save the CSV file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Target types with their characteristics
    target_types = {
        'vehicle': {'speed_range': (5, 40), 'heading_variance': 5, 'stop_probability': 0.05},
        'infantry': {'speed_range': (1, 6), 'heading_variance': 15, 'stop_probability': 0.1},
        'artillery': {'speed_range': (3, 15), 'heading_variance': 8, 'stop_probability': 0.2},
        'command': {'speed_range': (4, 25), 'heading_variance': 10, 'stop_probability': 0.15}
    }
    
    # Calculate total observations
    observations_per_day = 24 * 60 // sample_interval_minutes
    total_observations = num_targets * observations_per_day * days
    print(f"Generating {total_observations} observations for {num_targets} targets over {days} days...")
    
    # Create timestamps
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)
    timestamps = [start_time + datetime.timedelta(minutes=i*sample_interval_minutes) for i in range(observations_per_day * days)]
    
    # Define geographical area (modify this to match your study area)
    # These coordinates can be adjusted based on where your elevation/land use data is centered
    area = {
        'lat_min': 34.0, 'lat_max': 34.2,
        'lon_min': -118.2, 'lon_max': -118.0
    }
    
    # Generate data in chunks to avoid memory issues
    data = []
    target_state = {}  # Store current state for each target
    
    # Initialize targets
    for target_id in range(1, num_targets+1):
        target_class = list(target_types.keys())[target_id % len(target_types)]
        target_state[target_id] = {
            'latitude': np.random.uniform(area['lat_min'], area['lat_max']),
            'longitude': np.random.uniform(area['lon_min'], area['lon_max']),
            'heading': np.random.uniform(0, 360),
            'speed': np.random.uniform(*target_types[target_class]['speed_range']),
            'target_class': target_class,
            'movement_state': 'moving',  # or 'stopped'
            'stop_duration': 0,
            'formation_id': target_id % 20,  # Group targets into formations
            'last_terrain_cost': np.random.uniform(0.2, 0.8)
        }
    
    # Generate observations
    for i, timestamp in tqdm(enumerate(timestamps), total=len(timestamps)):
        for target_id, state in target_state.items():
            target_class = state['target_class']
            characteristics = target_types[target_class]
            
            # Time of day effects
            is_night = 1 if timestamp.hour >= 18 or timestamp.hour <= 6 else 0
            hour = timestamp.hour
            
            # Movement state transitions
            if state['movement_state'] == 'moving':
                # Check if target stops
                if np.random.random() < characteristics['stop_probability']:
                    state['movement_state'] = 'stopped'
                    state['stop_duration'] = np.random.randint(3, 24)  # periods to remain stopped
                    state['speed'] = 0
            else:  # stopped
                if state['stop_duration'] > 0:
                    state['stop_duration'] -= 1
                else:
                    state['movement_state'] = 'moving'
                    state['speed'] = np.random.uniform(*characteristics['speed_range'])
                    # Possibly change heading significantly when resuming movement
                    if np.random.random() < 0.3:
                        state['heading'] = (state['heading'] + np.random.uniform(-90, 90)) % 360
            
            if state['movement_state'] == 'moving':
                # Adjust heading based on terrain (simulated)
                # In a real system, this would use actual terrain data
                terrain_influence = np.random.normal(0, 20) * state['last_terrain_cost']
                state['heading'] = (state['heading'] + terrain_influence) % 360
                
                # Add random variation to heading
                heading_change = np.random.normal(0, characteristics['heading_variance'])
                state['heading'] = (state['heading'] + heading_change) % 360
                
                # Add random variation to speed
                speed_change = np.random.normal(0, characteristics['speed_range'][1] * 0.05)
                state['speed'] = max(0, min(characteristics['speed_range'][1] * 1.2, 
                                           state['speed'] + speed_change))
                
                # Night time speed reduction
                if is_night:
                    state['speed'] *= 0.8
                
                # Calculate new position
                distance = state['speed'] * (sample_interval_minutes/60)  # convert to hours
                lat_change = distance * np.cos(np.radians(state['heading'])) / 111
                lon_change = distance * np.sin(np.radians(state['heading'])) / (111 * np.cos(np.radians(state['latitude'])))
                
                state['latitude'] += lat_change
                state['longitude'] += lon_change
                
                # Ensure target stays within bounds, bouncing off the edges
                if state['latitude'] < area['lat_min']:
                    state['latitude'] = 2 * area['lat_min'] - state['latitude']
                    state['heading'] = (180 - state['heading']) % 360
                elif state['latitude'] > area['lat_max']:
                    state['latitude'] = 2 * area['lat_max'] - state['latitude']
                    state['heading'] = (180 - state['heading']) % 360
                
                if state['longitude'] < area['lon_min']:
                    state['longitude'] = 2 * area['lon_min'] - state['longitude']
                    state['heading'] = (360 - state['heading']) % 360
                elif state['longitude'] > area['lon_max']:
                    state['longitude'] = 2 * area['lon_max'] - state['longitude']
                    state['heading'] = (360 - state['heading']) % 360
                
                # Update simulated terrain cost
                state['last_terrain_cost'] = min(0.9, max(0.1, state['last_terrain_cost'] + np.random.normal(0, 0.1)))
            
            # Add formation variation for targets in the same formation
            formation_dispersion = 0
            if target_id % 20 > 0:  # Not the lead element
                # Add some formation-based offsets
                formation_dispersion = np.random.uniform(0.1, 0.5)
            
            # Add observation to dataset
            data.append({
                'target_id': f'T{target_id:03d}',
                'timestamp': timestamp,
                'latitude': state['latitude'],
                'longitude': state['longitude'],
                'target_class': state['target_class'],
                'speed': state['speed'],
                'heading': state['heading'],
                'is_night': is_night,
                'hour': hour,
                'formation_id': f'F{state["formation_id"]:02d}',
                'formation_dispersion': formation_dispersion,
                'simulated_terrain_cost': state['last_terrain_cost']
            })
        
        # Write data in chunks to avoid memory issues
        if len(data) > 1000000:
            if i == 0:  # First chunk, create new file
                pd.DataFrame(data).to_csv(output_file, index=False)
            else:  # Append to existing file
                pd.DataFrame(data).to_csv(output_file, mode='a', header=False, index=False)
            data = []
    
    # Write any remaining data
    if data:
        if os.path.exists(output_file):
            pd.DataFrame(data).to_csv(output_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame(data).to_csv(output_file, index=False)
    
    print(f"Dataset generated and saved to {output_file}")
    print(f"Total size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Generate a few blue forces as well
    generate_blue_forces_data(area, timestamps, output_file.replace('target_data.csv', 'blue_forces.csv'))

def generate_blue_forces_data(area, timestamps, output_file):
    """Generate blue forces data"""
    num_blue = 15  # Number of blue force units
    data = []
    blue_state = {}
    
    # Initialize blue forces
    for blue_id in range(1, num_blue+1):
        blue_state[blue_id] = {
            'latitude': np.random.uniform(area['lat_min'], area['lat_max']),
            'longitude': np.random.uniform(area['lon_min'], area['lon_max']),
            'heading': np.random.uniform(0, 360),
            'speed': np.random.uniform(3, 25),
            'blue_type': np.random.choice(['recon', 'infantry', 'vehicle', 'command']),
            'movement_state': 'moving'
        }
    
    # Generate at 1/10th the frequency of target data
    sampled_timestamps = timestamps[::10]
    
    for timestamp in tqdm(sampled_timestamps, desc="Generating blue forces"):
        for blue_id, state in blue_state.items():
            # Simple movement logic
            if np.random.random() < 0.1:
                state['heading'] = (state['heading'] + np.random.uniform(-45, 45)) % 360
                state['speed'] = max(0, np.random.uniform(0, 30))
            
            # Calculate new position
            if state['speed'] > 0:
                time_diff = 10 / 60  # 10 minute intervals in hours
                distance = state['speed'] * time_diff
                lat_change = distance * np.cos(np.radians(state['heading'])) / 111
                lon_change = distance * np.sin(np.radians(state['heading'])) / (111 * np.cos(np.radians(state['latitude'])))
                
                state['latitude'] += lat_change
                state['longitude'] += lon_change
                
                # Ensure blue force stays within bounds
                state['latitude'] = min(max(state['latitude'], area['lat_min']), area['lat_max'])
                state['longitude'] = min(max(state['longitude'], area['lon_min']), area['lon_max'])
            
            # Add to dataset
            data.append({
                'blue_id': f'B{blue_id:03d}',
                'timestamp': timestamp,
                'latitude': state['latitude'],
                'longitude': state['longitude'],
                'blue_type': state['blue_type'],
                'speed': state['speed'],
                'heading': state['heading']
            })
    
    pd.DataFrame(data).to_csv(output_file, index=False)
    print(f"Blue forces data saved to {output_file}")

def generate_sample_terrain_data(output_dir="data"):
    """Generate simple sample terrain data if none exists"""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define area
    lat_min, lat_max = 34.0, 34.2
    lon_min, lon_max = -118.2, -118.0
    
    # Grid resolution
    resolution = 100
    
    # Create grid points
    lat_vals = np.linspace(lat_min, lat_max, resolution)
    lon_vals = np.linspace(lon_min, lon_max, resolution)
    
    # Generate elevation data
    elevation_data = []
    for lat in lat_vals:
        for lon in lon_vals:
            # Create synthetic elevation using perlin noise (simplified here)
            base_elevation = 100 + 500 * (np.sin(lat * 100) * np.cos(lon * 100) + 
                                       np.sin(lat * 50) * np.cos(lon * 50) * 0.5)
            
            # Add some random variation
            elevation = base_elevation + np.random.normal(0, 20)
            
            elevation_data.append({
                'latitude': lat,
                'longitude': lon,
                'elevation': elevation
            })
    
    # Save elevation data
    pd.DataFrame(elevation_data).to_csv(f"{output_dir}/elevation_map.csv", index=False)
    print(f"Sample elevation data saved to {output_dir}/elevation_map.csv")
    
    # Generate land use data
    land_use_types = ['road', 'urban', 'forest', 'open_ground', 'light_vegetation', 'water']
    land_use_data = []
    
    for lat in lat_vals:
        for lon in lon_vals:
            # Simple heuristic to determine land use
            x_norm = (lon - lon_min) / (lon_max - lon_min)
            y_norm = (lat - lat_min) / (lat_max - lat_min)
            
            # Create patterns for different land use types
            if (x_norm * 10) % 1 < 0.1 or (y_norm * 10) % 1 < 0.1:
                land_type = 'road'  # Grid of roads
            elif np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2) < 0.2:
                land_type = 'urban'  # Center is urban
            elif np.sin(x_norm * 30) * np.cos(y_norm * 30) > 0.7:
                land_type = 'forest'  # Forest patches
            elif np.sin(x_norm * 20) * np.cos(y_norm * 20) < -0.8:
                land_type = 'water'  # Some water features
            elif np.random.random() < 0.3:
                land_type = 'light_vegetation'
            else:
                land_type = 'open_ground'
            
            land_use_data.append({
                'latitude': lat,
                'longitude': lon,
                'land_use_type': land_type,
                'concealment': 0.9 if land_type == 'forest' else 
                            0.7 if land_type == 'urban' else
                            0.5 if land_type == 'light_vegetation' else 0.1
            })
    
    # Save land use data
    pd.DataFrame(land_use_data).to_csv(f"{output_dir}/land_use.csv", index=False)
    print(f"Sample land use data saved to {output_dir}/land_use.csv")

if __name__ == "__main__":
    # Adjust parameters as needed
    generate_large_target_dataset(
        num_targets=100,
        days=14,
        sample_interval_minutes=5,
        output_file="data/target_data.csv"
    )
    
    # Generate sample terrain data if needed
    generate_sample_terrain_data()