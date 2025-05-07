"""
Evaluation Module

This module provides comprehensive evaluation capabilities for the
predictive tracking system, with a focus on measuring the accuracy
of predictions against ground truth data.

Key capabilities:
- Prediction accuracy metrics calculation
- Time-based accuracy degradation analysis
- Comparative evaluation of multiple prediction methods
- Error distribution visualization
- Confidence region evaluation
- Target-type specific performance assessment
- Terrain-based accuracy analysis
"""

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import datetime
import os
import math
import warnings
warnings.filterwarnings('ignore')

class PredictionEvaluator:
    """
    Class for evaluating prediction accuracy and performance.
    
    This class provides methods to evaluate the accuracy of target movement
    predictions against ground truth data using various metrics.
    
    Attributes:
        targets_df (pd.DataFrame): DataFrame containing ground truth target observations
        predictions (dict): Dictionary of prediction results
        evaluation_results (dict): Results of evaluation metrics
        config (dict): Configuration parameters
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, targets_df, predictions=None, config=None, verbose=True):
        """
        Initialize the PredictionEvaluator class.
        
        Args:
            targets_df (pd.DataFrame): DataFrame containing ground truth target observations
            predictions (dict): Dictionary of prediction results (optional)
            config (dict): Configuration parameters (optional)
            verbose (bool): Whether to print detailed information
        """
        self.targets_df = targets_df
        self.predictions = predictions if predictions is not None else {}
        self.evaluation_results = {}
        self.verbose = verbose
        
        # Default configuration
        self.default_config = {
            'error_metrics': ['mae', 'rmse', 'circular_error', 'hausdorff'],  # Metrics to calculate
            'time_horizons': [5, 10, 15, 30, 60],  # Time horizons to evaluate (minutes)
            'distance_thresholds': [50, 100, 200, 500],  # Distance thresholds for binary accuracy (meters)
            'confidence_levels': [0.5, 0.7, 0.9],  # Confidence levels for region evaluation
            'output_folder': './output',  # Folder for saving visualizations
            'cmap': 'viridis',  # Default colormap for visualizations
            'dpi': 150,  # DPI for saved figures
            'meters_per_degree': None,  # Custom conversion from degrees to meters (optional)
            'exclude_stationary': True,  # Whether to exclude stationary targets from evaluation
            'min_speed_threshold': 0.5,  # Minimum speed (km/h) to consider target as moving
            'time_weight_factor': 0.95,  # Weight factor for time-based accuracy calculation
            'error_scaling': 'linear',  # Scaling of error ('linear', 'logarithmic', 'quadratic')
            'segmentation': {  # Segmentation for conditional evaluation
                'terrain_types': ['urban', 'forest', 'open', 'road', 'water'],
                'target_classes': ['vehicle', 'infantry', 'artillery', 'command'],
                'time_of_day': ['day', 'night'],
                'weather_conditions': ['clear', 'rain', 'fog', 'snow']
            }
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Create output folder if it doesn't exist
        if self.config['output_folder']:
            os.makedirs(self.config['output_folder'], exist_ok=True)
        
        # Initialize results storage
        self.reset_results()
        
        if self.verbose:
            print("Prediction Evaluator initialized")
    
    def reset_results(self):
        """
        Reset evaluation results.
        
        Returns:
            None
        """
        self.evaluation_results = {
            'overall': {},
            'by_target': {},
            'by_time_horizon': {},
            'by_method': {},
            'by_target_class': {},
            'by_terrain_type': {},
            'by_time_of_day': {},
            'by_weather': {},
            'confidence_regions': {},
            'time_degradation': {}
        }
    
    def evaluate_predictions(self, predictions=None, metrics=None, time_horizons=None):
        """
        Evaluate predictions using specified metrics.
        
        Args:
            predictions (dict): Dictionary of prediction results (optional)
            metrics (list): List of metrics to calculate (optional)
            time_horizons (list): List of time horizons to evaluate (optional)
            
        Returns:
            dict: Evaluation results
        """
        # Update predictions if provided
        if predictions is not None:
            self.predictions = predictions
        
        # Use specified or default metrics
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Use specified or default time horizons
        time_horizons = time_horizons if time_horizons is not None else self.config['time_horizons']
        
        # Reset results
        self.reset_results()
        
        # Overall evaluation
        self.evaluate_overall(metrics)
        
        # Evaluation by time horizon
        for horizon in time_horizons:
            self.evaluate_by_time_horizon(horizon, metrics)
        
        # Evaluation by method (if multiple methods)
        methods = self._get_unique_methods()
        for method in methods:
            self.evaluate_by_method(method, metrics)
        
        # Evaluation by target class (if available)
        if 'target_class' in self.targets_df.columns:
            target_classes = self.targets_df['target_class'].unique()
            for target_class in target_classes:
                self.evaluate_by_target_class(target_class, metrics)
        
        # Evaluation by target
        targets = self.targets_df['target_id'].unique()
        for target_id in targets:
            self.evaluate_by_target(target_id, metrics)
        
        # Evaluation of confidence regions (if available)
        if any('confidence_regions' in pred for pred in self.predictions.values() if isinstance(pred, dict)):
            self.evaluate_confidence_regions()
        
        # Evaluate time-based accuracy degradation
        self.evaluate_time_degradation(metrics)
        
        if self.verbose:
            print(f"Completed evaluation with {len(metrics)} metrics across {len(time_horizons)} time horizons")
        
        return self.evaluation_results
    
    def evaluate_overall(self, metrics=None):
        """
        Evaluate overall prediction accuracy.
        
        Args:
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Overall evaluation results
        """
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Calculate error metrics for all predictions
        errors = self._calculate_prediction_errors()
        
        if errors:
            # Calculate each requested metric
            for metric in metrics:
                if metric == 'mae':
                    # Mean Absolute Error
                    results['mae'] = np.mean([e['error_distance'] for e in errors])
                
                elif metric == 'rmse':
                    # Root Mean Squared Error
                    results['rmse'] = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                
                elif metric == 'circular_error':
                    # Circular Error Probable (CEP) - radius containing 50% of errors
                    error_distances = [e['error_distance'] for e in errors]
                    results['cep50'] = np.percentile(error_distances, 50)
                    results['cep90'] = np.percentile(error_distances, 90)
                
                elif metric == 'hausdorff':
                    # Mean Hausdorff Distance for path predictions
                    path_errors = [e for e in errors if 'path_error' in e]
                    if path_errors:
                        results['hausdorff'] = np.mean([e['path_error'] for e in path_errors])
            
            # Calculate binary accuracy metrics for different distance thresholds
            for threshold in self.config['distance_thresholds']:
                correct = sum(1 for e in errors if e['error_distance'] <= threshold)
                total = len(errors)
                results[f'accuracy_{threshold}m'] = correct / total if total > 0 else 0
            
            # Store results
            self.evaluation_results['overall'] = results
            
            if self.verbose:
                print("Overall evaluation results:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        return results
    
    def evaluate_by_time_horizon(self, time_horizon, metrics=None):
        """
        Evaluate prediction accuracy for a specific time horizon.
        
        Args:
            time_horizon (int): Time horizon in minutes
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Evaluation results for the time horizon
        """
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Filter predictions by time horizon
        horizon_predictions = {
            k: p for k, p in self.predictions.items() 
            if isinstance(p, dict) and 'minutes_ahead' in p and p['minutes_ahead'] == time_horizon
        }
        
        # Calculate error metrics for filtered predictions
        errors = self._calculate_prediction_errors(horizon_predictions)
        
        if errors:
            # Calculate each requested metric
            for metric in metrics:
                if metric == 'mae':
                    # Mean Absolute Error
                    results['mae'] = np.mean([e['error_distance'] for e in errors])
                
                elif metric == 'rmse':
                    # Root Mean Squared Error
                    results['rmse'] = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                
                elif metric == 'circular_error':
                    # Circular Error Probable (CEP) - radius containing 50% of errors
                    error_distances = [e['error_distance'] for e in errors]
                    results['cep50'] = np.percentile(error_distances, 50)
                    results['cep90'] = np.percentile(error_distances, 90)
                
                elif metric == 'hausdorff':
                    # Mean Hausdorff Distance for path predictions
                    path_errors = [e for e in errors if 'path_error' in e]
                    if path_errors:
                        results['hausdorff'] = np.mean([e['path_error'] for e in path_errors])
            
            # Calculate binary accuracy metrics for different distance thresholds
            for threshold in self.config['distance_thresholds']:
                correct = sum(1 for e in errors if e['error_distance'] <= threshold)
                total = len(errors)
                results[f'accuracy_{threshold}m'] = correct / total if total > 0 else 0
            
            # Store results
            self.evaluation_results['by_time_horizon'][time_horizon] = results
            
            if self.verbose:
                print(f"Evaluation results for time horizon {time_horizon} minutes:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        return results
    
    def evaluate_by_method(self, method, metrics=None):
        """
        Evaluate prediction accuracy for a specific method.
        
        Args:
            method (str): Prediction method name
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Evaluation results for the method
        """
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Filter predictions by method
        method_predictions = {
            k: p for k, p in self.predictions.items() 
            if isinstance(p, dict) and 'method' in p and p['method'] == method
        }
        
        # Calculate error metrics for filtered predictions
        errors = self._calculate_prediction_errors(method_predictions)
        
        if errors:
            # Calculate each requested metric
            for metric in metrics:
                if metric == 'mae':
                    # Mean Absolute Error
                    results['mae'] = np.mean([e['error_distance'] for e in errors])
                
                elif metric == 'rmse':
                    # Root Mean Squared Error
                    results['rmse'] = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                
                elif metric == 'circular_error':
                    # Circular Error Probable (CEP) - radius containing 50% of errors
                    error_distances = [e['error_distance'] for e in errors]
                    results['cep50'] = np.percentile(error_distances, 50)
                    results['cep90'] = np.percentile(error_distances, 90)
                
                elif metric == 'hausdorff':
                    # Mean Hausdorff Distance for path predictions
                    path_errors = [e for e in errors if 'path_error' in e]
                    if path_errors:
                        results['hausdorff'] = np.mean([e['path_error'] for e in path_errors])
            
            # Calculate binary accuracy metrics for different distance thresholds
            for threshold in self.config['distance_thresholds']:
                correct = sum(1 for e in errors if e['error_distance'] <= threshold)
                total = len(errors)
                results[f'accuracy_{threshold}m'] = correct / total if total > 0 else 0
            
            # Store results
            self.evaluation_results['by_method'][method] = results
            
            if self.verbose:
                print(f"Evaluation results for method {method}:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        return results
    
    def evaluate_by_target_class(self, target_class, metrics=None):
        """
        Evaluate prediction accuracy for a specific target class.
        
        Args:
            target_class (str): Target class name
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Evaluation results for the target class
        """
        if 'target_class' not in self.targets_df.columns:
            if self.verbose:
                print("Target class information not available in targets_df")
            return None
        
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Get target IDs for the specified class
        target_ids = self.targets_df[self.targets_df['target_class'] == target_class]['target_id'].unique()
        
        # Filter predictions by target IDs
        class_predictions = {
            k: p for k, p in self.predictions.items() 
            if isinstance(p, dict) and 'target_id' in p and p['target_id'] in target_ids
        }
        
        # Calculate error metrics for filtered predictions
        errors = self._calculate_prediction_errors(class_predictions)
        
        if errors:
            # Calculate each requested metric
            for metric in metrics:
                if metric == 'mae':
                    # Mean Absolute Error
                    results['mae'] = np.mean([e['error_distance'] for e in errors])
                
                elif metric == 'rmse':
                    # Root Mean Squared Error
                    results['rmse'] = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                
                elif metric == 'circular_error':
                    # Circular Error Probable (CEP) - radius containing 50% of errors
                    error_distances = [e['error_distance'] for e in errors]
                    results['cep50'] = np.percentile(error_distances, 50)
                    results['cep90'] = np.percentile(error_distances, 90)
                
                elif metric == 'hausdorff':
                    # Mean Hausdorff Distance for path predictions
                    path_errors = [e for e in errors if 'path_error' in e]
                    if path_errors:
                        results['hausdorff'] = np.mean([e['path_error'] for e in path_errors])
            
            # Calculate binary accuracy metrics for different distance thresholds
            for threshold in self.config['distance_thresholds']:
                correct = sum(1 for e in errors if e['error_distance'] <= threshold)
                total = len(errors)
                results[f'accuracy_{threshold}m'] = correct / total if total > 0 else 0
            
            # Store results
            self.evaluation_results['by_target_class'][target_class] = results
            
            if self.verbose:
                print(f"Evaluation results for target class {target_class}:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        return results
    
    def evaluate_by_target(self, target_id, metrics=None):
        """
        Evaluate prediction accuracy for a specific target.
        
        Args:
            target_id: Target ID
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Evaluation results for the target
        """
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Filter predictions by target ID
        target_predictions = {
            k: p for k, p in self.predictions.items() 
            if isinstance(p, dict) and 'target_id' in p and p['target_id'] == target_id
        }
        
        # Calculate error metrics for filtered predictions
        errors = self._calculate_prediction_errors(target_predictions)
        
        if errors:
            # Calculate each requested metric
            for metric in metrics:
                if metric == 'mae':
                    # Mean Absolute Error
                    results['mae'] = np.mean([e['error_distance'] for e in errors])
                
                elif metric == 'rmse':
                    # Root Mean Squared Error
                    results['rmse'] = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                
                elif metric == 'circular_error':
                    # Circular Error Probable (CEP) - radius containing 50% of errors
                    error_distances = [e['error_distance'] for e in errors]
                    results['cep50'] = np.percentile(error_distances, 50)
                    results['cep90'] = np.percentile(error_distances, 90)
                
                elif metric == 'hausdorff':
                    # Mean Hausdorff Distance for path predictions
                    path_errors = [e for e in errors if 'path_error' in e]
                    if path_errors:
                        results['hausdorff'] = np.mean([e['path_error'] for e in path_errors])
            
            # Calculate binary accuracy metrics for different distance thresholds
            for threshold in self.config['distance_thresholds']:
                correct = sum(1 for e in errors if e['error_distance'] <= threshold)
                total = len(errors)
                results[f'accuracy_{threshold}m'] = correct / total if total > 0 else 0
            
            # Store results
            self.evaluation_results['by_target'][target_id] = results
            
            if self.verbose:
                print(f"Evaluation results for target {target_id}:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        return results
    
    def evaluate_by_terrain_type(self, terrain_type, metrics=None):
        """
        Evaluate prediction accuracy for a specific terrain type.
        
        Args:
            terrain_type (str): Terrain type name
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Evaluation results for the terrain type
        """
        if 'land_use_type' not in self.targets_df.columns:
            if self.verbose:
                print("Terrain type information not available in targets_df")
            return None
        
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Calculate error metrics for all predictions
        all_errors = self._calculate_prediction_errors()
        
        # Filter errors by terrain type
        terrain_errors = [
            e for e in all_errors 
            if 'terrain_type' in e and e['terrain_type'] == terrain_type
        ]
        
        if terrain_errors:
            # Calculate each requested metric
            for metric in metrics:
                if metric == 'mae':
                    # Mean Absolute Error
                    results['mae'] = np.mean([e['error_distance'] for e in terrain_errors])
                
                elif metric == 'rmse':
                    # Root Mean Squared Error
                    results['rmse'] = np.sqrt(np.mean([e['error_distance']**2 for e in terrain_errors]))
                
                elif metric == 'circular_error':
                    # Circular Error Probable (CEP) - radius containing 50% of errors
                    error_distances = [e['error_distance'] for e in terrain_errors]
                    results['cep50'] = np.percentile(error_distances, 50)
                    results['cep90'] = np.percentile(error_distances, 90)
                
                elif metric == 'hausdorff':
                    # Mean Hausdorff Distance for path predictions
                    path_errors = [e for e in terrain_errors if 'path_error' in e]
                    if path_errors:
                        results['hausdorff'] = np.mean([e['path_error'] for e in path_errors])
            
            # Calculate binary accuracy metrics for different distance thresholds
            for threshold in self.config['distance_thresholds']:
                correct = sum(1 for e in terrain_errors if e['error_distance'] <= threshold)
                total = len(terrain_errors)
                results[f'accuracy_{threshold}m'] = correct / total if total > 0 else 0
            
            # Store results
            self.evaluation_results['by_terrain_type'][terrain_type] = results
            
            if self.verbose:
                print(f"Evaluation results for terrain type {terrain_type}:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        return results
    
    def evaluate_confidence_regions(self):
        """
        Evaluate if ground truth positions fall within predicted confidence regions.
        
        Returns:
            dict: Confidence region evaluation results
        """
        results = {}
        
        # Get predictions with confidence regions
        confidence_predictions = {
            k: p for k, p in self.predictions.items() 
            if isinstance(p, dict) and 'confidence_regions' in p
        }
        
        if confidence_predictions:
            # Initialize results for each confidence level
            for level in self.config['confidence_levels']:
                results[level] = {
                    'count': 0,
                    'correct': 0,
                    'accuracy': 0.0,
                    'avg_area': 0.0
                }
            
            # Evaluate each prediction
            for pred_key, prediction in confidence_predictions.items():
                confidence_regions = prediction['confidence_regions']
                
                # Get target ID and time horizon
                target_id = prediction.get('target_id')
                time_horizon = prediction.get('minutes_ahead')
                
                if target_id is None or time_horizon is None:
                    continue
                
                # Get ground truth position at prediction time
                prediction_time = self._get_prediction_time(prediction)
                if prediction_time is None:
                    continue
                
                ground_truth = self._get_ground_truth(target_id, prediction_time)
                if ground_truth is None:
                    continue
                
                # Check if ground truth falls within each confidence region
                for level_str, region in confidence_regions.items():
                    try:
                        # Convert level string to float (e.g., '90' -> 0.9)
                        level = float(level_str) / 100
                        
                        if level not in results:
                            # Skip levels not in config
                            continue
                        
                        results[level]['count'] += 1
                        
                        # Check if point is inside confidence region
                        is_inside = self._point_in_region(
                            (ground_truth['longitude'], ground_truth['latitude']),
                            region
                        )
                        
                        if is_inside:
                            results[level]['correct'] += 1
                        
                        # Add region area if available
                        if 'area_km2' in region:
                            results[level]['avg_area'] += region['area_km2']
                    except (ValueError, KeyError):
                        # Skip invalid levels
                        continue
            
            # Calculate accuracy and average area
            for level in self.config['confidence_levels']:
                if level in results:
                    if results[level]['count'] > 0:
                        results[level]['accuracy'] = results[level]['correct'] / results[level]['count']
                        results[level]['avg_area'] /= results[level]['count']
            
            # Store results
            self.evaluation_results['confidence_regions'] = results
            
            if self.verbose:
                print("Confidence region evaluation results:")
                for level, metrics in results.items():
                    if metrics['count'] > 0:
                        print(f"  {level*100:.0f}% confidence: "
                              f"{metrics['accuracy']*100:.1f}% accuracy, "
                              f"avg area {metrics['avg_area']:.2f} km²")
        
        return results
    
    def evaluate_time_degradation(self, metrics=None):
        """
        Evaluate how prediction accuracy degrades over time.
        
        Args:
            metrics (list): List of metrics to calculate (optional)
            
        Returns:
            dict: Time degradation evaluation results
        """
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Calculate each metric
        results = {}
        
        # Group predictions by time horizon
        time_horizons = []
        for pred in self.predictions.values():
            if isinstance(pred, dict) and 'minutes_ahead' in pred:
                time_horizons.append(pred['minutes_ahead'])
        
        time_horizons = sorted(set(time_horizons))
        
        if not time_horizons:
            return results
        
        # Calculate metrics for each time horizon
        for metric in metrics:
            results[metric] = {}
            
            for horizon in time_horizons:
                # Get metric value for this time horizon
                if horizon in self.evaluation_results['by_time_horizon']:
                    horizon_results = self.evaluation_results['by_time_horizon'][horizon]
                    if metric in horizon_results:
                        results[metric][horizon] = horizon_results[metric]
                    elif metric == 'circular_error' and 'cep50' in horizon_results:
                        # Use CEP50 for circular error
                        results[metric][horizon] = horizon_results['cep50']
                    elif f'accuracy_{self.config["distance_thresholds"][0]}m' in horizon_results:
                        # Use first threshold accuracy if specific metric not found
                        threshold = self.config["distance_thresholds"][0]
                        results[metric][horizon] = horizon_results[f'accuracy_{threshold}m']
        
        # Store results
        self.evaluation_results['time_degradation'] = results
        
        if self.verbose:
            print("Time degradation evaluation results:")
            for metric, values in results.items():
                print(f"  {metric}: {', '.join([f'{h}min: {v:.2f}' for h, v in values.items()])}")
        
        return results
    
    def compare_methods(self, methods=None, metrics=None, time_horizons=None):
        """
        Compare the performance of multiple prediction methods.
        
        Args:
            methods (list): List of methods to compare (optional)
            metrics (list): List of metrics to use for comparison (optional)
            time_horizons (list): List of time horizons to compare (optional)
            
        Returns:
            pd.DataFrame: Comparison results
        """
        # Use all available methods if not specified
        if methods is None:
            methods = self._get_unique_methods()
        
        # Use specified or default metrics
        metrics = metrics if metrics is not None else self.config['error_metrics']
        
        # Use specified or default time horizons
        time_horizons = time_horizons if time_horizons is not None else self.config['time_horizons']
        
        # Initialize results
        comparison = {}
        
        # Generate comparison data
        for method in methods:
            comparison[method] = {}
            
            if method in self.evaluation_results['by_method']:
                method_results = self.evaluation_results['by_method'][method]
                
                # Overall results
                for metric in metrics:
                    if metric in method_results:
                        comparison[method][f'overall_{metric}'] = method_results[metric]
                    elif metric == 'circular_error' and 'cep50' in method_results:
                        # Use CEP50 for circular error
                        comparison[method][f'overall_{metric}'] = method_results['cep50']
                
                # Results by time horizon
                for horizon in time_horizons:
                    # Filter predictions by method and time horizon
                    horizon_predictions = {
                        k: p for k, p in self.predictions.items() 
                        if isinstance(p, dict) and 'method' in p and p['method'] == method
                        and 'minutes_ahead' in p and p['minutes_ahead'] == horizon
                    }
                    
                    if horizon_predictions:
                        # Calculate errors
                        errors = self._calculate_prediction_errors(horizon_predictions)
                        
                        if errors:
                            # Calculate each metric
                            for metric in metrics:
                                if metric == 'mae':
                                    # Mean Absolute Error
                                    comparison[method][f'{horizon}min_{metric}'] = np.mean([e['error_distance'] for e in errors])
                                
                                elif metric == 'rmse':
                                    # Root Mean Squared Error
                                    comparison[method][f'{horizon}min_{metric}'] = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                                
                                elif metric == 'circular_error':
                                    # Circular Error Probable (CEP) - radius containing 50% of errors
                                    error_distances = [e['error_distance'] for e in errors]
                                    comparison[method][f'{horizon}min_cep50'] = np.percentile(error_distances, 50)
                                
                                elif metric == 'hausdorff':
                                    # Mean Hausdorff Distance for path predictions
                                    path_errors = [e for e in errors if 'path_error' in e]
                                    if path_errors:
                                        comparison[method][f'{horizon}min_{metric}'] = np.mean([e['path_error'] for e in path_errors])
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        return comparison_df
    
    def calculate_performance_score(self, weights=None):
        """
        Calculate an overall performance score based on weighted metrics.
        
        Args:
            weights (dict): Weights for different metrics (optional)
            
        Returns:
            dict: Performance scores for each method
        """
        # Default weights
        default_weights = {
            'mae': 0.3,
            'cep50': 0.3,
            'accuracy_100m': 0.4
        }
        
        # Use provided weights or defaults
        weights = weights if weights is not None else default_weights
        
        # Get all methods
        methods = self._get_unique_methods()
        
        # Calculate scores
        scores = {}
        
        for method in methods:
            if method in self.evaluation_results['by_method']:
                method_results = self.evaluation_results['by_method'][method]
                
                # Initialize score components
                score_components = {}
                total_weight = 0
                
                # Calculate weighted score components
                for metric, weight in weights.items():
                    if metric in method_results:
                        # For error metrics (lower is better), invert the score
                        if metric in ['mae', 'rmse', 'cep50', 'cep90', 'hausdorff']:
                            # Normalize to [0, 1] using a reference value
                            reference = 2 * method_results[metric]  # Use 2x the value as reference
                            normalized = max(0, 1 - method_results[metric] / reference)
                            score_components[metric] = normalized * weight
                        else:
                            # For accuracy metrics (higher is better), use directly
                            score_components[metric] = method_results[metric] * weight
                        
                        total_weight += weight
                
                # Calculate total score
                if total_weight > 0:
                    scores[method] = sum(score_components.values()) / total_weight
                else:
                    scores[method] = 0
        
        return scores
    
    def plot_error_distribution(self, by_method=False, by_time_horizon=False, figsize=(10, 6), filename=None):
        """
        Plot distribution of prediction errors.
        
        Args:
            by_method (bool): Whether to separate by method
            by_time_horizon (bool): Whether to separate by time horizon
            figsize (tuple): Figure size
            filename (str): Output filename (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Calculate errors for all predictions
        errors = self._calculate_prediction_errors()
        
        if not errors:
            if self.verbose:
                print("No errors to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if by_method and not by_time_horizon:
            # Group by method
            methods = self._get_unique_methods()
            
            for method in methods:
                method_errors = [e['error_distance'] for e in errors 
                                if 'method' in e and e['method'] == method]
                
                if method_errors:
                    # Plot KDE
                    sns.kdeplot(method_errors, label=method, ax=ax)
            
            ax.set_title('Prediction Error Distribution by Method')
            ax.legend()
            
        elif by_time_horizon and not by_method:
            # Group by time horizon
            horizons = []
            for e in errors:
                if 'time_horizon' in e:
                    horizons.append(e['time_horizon'])
            
            horizons = sorted(set(horizons))
            
            for horizon in horizons:
                horizon_errors = [e['error_distance'] for e in errors 
                                 if 'time_horizon' in e and e['time_horizon'] == horizon]
                
                if horizon_errors:
                    # Plot KDE
                    sns.kdeplot(horizon_errors, label=f'{horizon} min', ax=ax)
            
            ax.set_title('Prediction Error Distribution by Time Horizon')
            ax.legend()
            
        elif by_method and by_time_horizon:
            # Group by method and time horizon
            methods = self._get_unique_methods()
            
            horizons = []
            for e in errors:
                if 'time_horizon' in e:
                    horizons.append(e['time_horizon'])
            
            horizons = sorted(set(horizons))
            
            # Create subplots for each horizon
            fig.clear()
            if len(horizons) > 2:
                ncols = min(3, len(horizons))
                nrows = (len(horizons) + ncols - 1) // ncols
            else:
                nrows, ncols = 1, len(horizons)
            
            axes = fig.subplots(nrows, ncols, sharex=True, sharey=True)
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            elif nrows == 1 or ncols == 1:
                axes = axes.reshape(-1)
            
            for i, horizon in enumerate(horizons):
                ax = axes.flat[i] if i < len(axes.flat) else None
                if ax is None:
                    break
                
                for method in methods:
                    method_horizon_errors = [e['error_distance'] for e in errors 
                                           if 'method' in e and e['method'] == method
                                           and 'time_horizon' in e and e['time_horizon'] == horizon]
                    
                    if method_horizon_errors:
                        # Plot KDE
                        sns.kdeplot(method_horizon_errors, label=method, ax=ax)
                
                ax.set_title(f'{horizon} min')
                if i == 0:
                    ax.legend()
            
            fig.suptitle('Prediction Error Distribution by Method and Time Horizon')
            fig.tight_layout()
            
        else:
            # Overall distribution
            error_distances = [e['error_distance'] for e in errors]
            
            # Plot KDE
            sns.kdeplot(error_distances, ax=ax)
            
            ax.set_title('Overall Prediction Error Distribution')
        
        # Set axis labels
        ax.set_xlabel('Error Distance (meters)')
        ax.set_ylabel('Density')
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_error_over_time(self, metric='mae', methods=None, figsize=(10, 6), filename=None):
        """
        Plot prediction error over different time horizons.
        
        Args:
            metric (str): Metric to plot
            methods (list): List of methods to include (optional)
            figsize (tuple): Figure size
            filename (str): Output filename (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Use specified or available methods
        if methods is None:
            methods = self._get_unique_methods()
        
        # Get time horizons
        time_horizons = sorted(self.evaluation_results['by_time_horizon'].keys())
        
        if not time_horizons:
            if self.verbose:
                print("No time horizons to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot error over time for each method
        for method in methods:
            if method in self.evaluation_results['by_method']:
                # Collect data points
                x = []
                y = []
                
                for horizon in time_horizons:
                    # Filter predictions by method and time horizon
                    horizon_predictions = {
                        k: p for k, p in self.predictions.items() 
                        if isinstance(p, dict) and 'method' in p and p['method'] == method
                        and 'minutes_ahead' in p and p['minutes_ahead'] == horizon
                    }
                    
                    if horizon_predictions:
                        # Calculate errors
                        errors = self._calculate_prediction_errors(horizon_predictions)
                        
                        if errors:
                            if metric == 'mae':
                                # Mean Absolute Error
                                value = np.mean([e['error_distance'] for e in errors])
                            elif metric == 'rmse':
                                # Root Mean Squared Error
                                value = np.sqrt(np.mean([e['error_distance']**2 for e in errors]))
                            elif metric == 'cep50':
                                # Circular Error Probable (50%)
                                error_distances = [e['error_distance'] for e in errors]
                                value = np.percentile(error_distances, 50)
                            elif metric == 'cep90':
                                # Circular Error Probable (90%)
                                error_distances = [e['error_distance'] for e in errors]
                                value = np.percentile(error_distances, 90)
                            elif metric.startswith('accuracy_'):
                                # Accuracy at distance threshold
                                threshold = int(metric.split('_')[1][:-1])  # Extract threshold (e.g., 100 from 'accuracy_100m')
                                correct = sum(1 for e in errors if e['error_distance'] <= threshold)
                                total = len(errors)
                                value = correct / total if total > 0 else 0
                            else:
                                continue
                            
                            x.append(horizon)
                            y.append(value)
                
                if x and y:
                    ax.plot(x, y, 'o-', label=method)
        
        # Set axis labels and title
        ax.set_xlabel('Time Horizon (minutes)')
        if metric.startswith('accuracy_'):
            ax.set_ylabel(f'Accuracy ({metric.split("_")[1]})')
            ax.set_title(f'Prediction Accuracy over Time Horizon')
        else:
            ax.set_ylabel(f'{metric.upper()} (meters)')
            ax.set_title(f'Prediction Error ({metric.upper()}) over Time Horizon')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_method_comparison(self, metrics=None, time_horizons=None, figsize=(12, 8), filename=None):
        """
        Plot comparative performance of different prediction methods.
        
        Args:
            metrics (list): List of metrics to include (optional)
            time_horizons (list): List of time horizons to include (optional)
            figsize (tuple): Figure size
            filename (str): Output filename (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Use specified or default metrics
        if metrics is None:
            metrics = ['mae', 'cep50', 'accuracy_100m']
        
        # Use specified or available time horizons
        if time_horizons is None:
            time_horizons = sorted(self.evaluation_results['by_time_horizon'].keys())
        
        # Get methods
        methods = self._get_unique_methods()
        
        if not methods:
            if self.verbose:
                print("No methods to compare")
            return None
        
        # Generate comparison data
        comparison = self.compare_methods(methods, metrics, time_horizons)
        
        if comparison.empty:
            if self.verbose:
                print("No comparison data available")
            return None
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot comparison for each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get columns for this metric
            if metric == 'circular_error':
                metric_cols = [col for col in comparison.columns if 'cep50' in col]
            else:
                metric_cols = [col for col in comparison.columns if metric in col]
            
            # Extract data
            data = comparison[metric_cols].copy()
            
            # Rename columns to show time horizons
            data.columns = [col.split('_')[0] if 'overall' in col else col.split('_')[0] for col in data.columns]
            
            # Transpose for plotting
            data = data.transpose()
            
            # Plot data
            data.plot(kind='bar', ax=ax)
            
            # Set title and labels
            if metric == 'circular_error':
                ax.set_title(f'Circular Error Probable (CEP50)')
                ax.set_ylabel('Meters')
            elif metric.startswith('accuracy_'):
                threshold = metric.split('_')[1]
                ax.set_title(f'Accuracy within {threshold}')
                ax.set_ylabel('Accuracy')
            else:
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel('Meters' if metric in ['mae', 'rmse'] else 'Value')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add value labels on bars
            for j, container in enumerate(ax.containers):
                ax.bar_label(container, fmt='%.1f', padding=3)
        
        # Set x-axis label
        axes[-1].set_xlabel('Time Horizon (minutes)')
        
        # Add legend
        axes[0].legend(title='Method')
        
        # Adjust layout
        fig.suptitle('Comparison of Prediction Methods')
        fig.tight_layout()
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_spatial_error(self, method=None, time_horizon=None, figsize=(10, 10), filename=None):
        """
        Plot spatial distribution of prediction errors on a map.
        
        Args:
            method (str): Method to plot (optional)
            time_horizon (int): Time horizon to plot (optional)
            figsize (tuple): Figure size
            filename (str): Output filename (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Calculate errors for all or filtered predictions
        if method is not None or time_horizon is not None:
            # Filter predictions
            filtered_predictions = {}
            
            for k, p in self.predictions.items():
                if not isinstance(p, dict):
                    continue
                
                if method is not None and ('method' not in p or p['method'] != method):
                    continue
                
                if time_horizon is not None and ('minutes_ahead' not in p or p['minutes_ahead'] != time_horizon):
                    continue
                
                filtered_predictions[k] = p
            
            errors = self._calculate_prediction_errors(filtered_predictions)
        else:
            errors = self._calculate_prediction_errors()
        
        if not errors:
            if self.verbose:
                print("No errors to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coordinates and error distances
        lons = [e['ground_truth']['longitude'] for e in errors]
        lats = [e['ground_truth']['latitude'] for e in errors]
        error_dists = [e['error_distance'] for e in errors]
        
        # Create scatter plot
        scatter = ax.scatter(lons, lats, c=error_dists, cmap='Reds', 
                           alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Error Distance (meters)')
        
        # Set axis labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        title = 'Spatial Distribution of Prediction Errors'
        if method is not None:
            title += f' - Method: {method}'
        if time_horizon is not None:
            title += f' - Time Horizon: {time_horizon} min'
        ax.set_title(title)
        
        # Set aspect ratio to equal for geographic data
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_confidence_region_reliability(self, figsize=(8, 6), filename=None):
        """
        Plot reliability diagram for confidence regions.
        
        Args:
            figsize (tuple): Figure size
            filename (str): Output filename (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        confidence_results = self.evaluation_results.get('confidence_regions', {})
        
        if not confidence_results:
            if self.verbose:
                print("No confidence region results to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        nominal_confidence = []
        actual_confidence = []
        areas = []
        
        for level, metrics in confidence_results.items():
            if metrics['count'] > 0:
                nominal_confidence.append(level)
                actual_confidence.append(metrics['accuracy'])
                areas.append(metrics['avg_area'])
        
        if not nominal_confidence:
            if self.verbose:
                print("No confidence region data points to plot")
            return None
        
        # Plot reference line (perfect calibration)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        # Plot confidence region reliability
        ax.plot(nominal_confidence, actual_confidence, 'o-', linewidth=2, label='Confidence Regions')
        
        # Add points with area information
        scatter = ax.scatter(nominal_confidence, actual_confidence, 
                           c=areas, cmap='viridis', s=100, 
                           edgecolors='black', linewidths=0.5, zorder=3)
        
        # Add colorbar for areas
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Average Region Area (km²)')
        
        # Set axis limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Nominal Confidence Level')
        ax.set_ylabel('Actual Confidence Level')
        ax.set_title('Confidence Region Reliability Diagram')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(self, output_file=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_file (str): Output filename (optional)
            
        Returns:
            str: Report text
        """
        # Create report
        report = []
        
        # Add title
        report.append("# Predictive Tracking System - Evaluation Report")
        report.append("")
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add overall results
        report.append("## Overall Performance")
        
        if self.evaluation_results['overall']:
            # Add table of metrics
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            
            for metric, value in self.evaluation_results['overall'].items():
                report.append(f"| {metric} | {value:.2f} |")
            
            report.append("")
        else:
            report.append("No overall evaluation results available.")
            report.append("")
        
        # Add method comparison
        methods = self._get_unique_methods()
        
        if methods and len(methods) > 1:
            report.append("## Method Comparison")
            
            # Calculate performance scores
            scores = self.calculate_performance_score()
            
            if scores:
                # Add table of scores
                report.append("### Performance Scores")
                report.append("| Method | Score |")
                report.append("|--------|-------|")
                
                for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"| {method} | {score:.3f} |")
                
                report.append("")
            
            # Add method comparison table
            report.append("### Key Metrics by Method")
            
            metrics = ['mae', 'rmse', 'cep50', 'accuracy_100m']
            available_metrics = []
            
            for metric in metrics:
                for method in methods:
                    if method in self.evaluation_results['by_method']:
                        method_results = self.evaluation_results['by_method'][method]
                        if metric in method_results:
                            if metric not in available_metrics:
                                available_metrics.append(metric)
            
            if available_metrics:
                # Create table header
                header = "| Method |"
                separator = "|--------|"
                
                for metric in available_metrics:
                    header += f" {metric.upper()} |"
                    separator += "--------|"
                
                report.append(header)
                report.append(separator)
                
                # Add data rows
                for method in methods:
                    if method in self.evaluation_results['by_method']:
                        method_results = self.evaluation_results['by_method'][method]
                        
                        row = f"| {method} |"
                        
                        for metric in available_metrics:
                            if metric in method_results:
                                row += f" {method_results[metric]:.2f} |"
                            else:
                                row += " - |"
                        
                        report.append(row)
                
                report.append("")
            else:
                report.append("No comparable metrics available for methods.")
                report.append("")
        
        # Add time horizon analysis
        time_horizons = sorted(self.evaluation_results['by_time_horizon'].keys())
        
        if time_horizons:
            report.append("## Prediction Accuracy by Time Horizon")
            
            # Create table header
            header = "| Time Horizon |"
            separator = "|-------------|"
            
            metrics = ['mae', 'rmse', 'cep50', 'accuracy_100m']
            available_metrics = []
            
            for metric in metrics:
                for horizon in time_horizons:
                    horizon_results = self.evaluation_results['by_time_horizon'].get(horizon, {})
                    if metric in horizon_results:
                        if metric not in available_metrics:
                            available_metrics.append(metric)
            
            for metric in available_metrics:
                header += f" {metric.upper()} |"
                separator += "--------|"
            
            report.append(header)
            report.append(separator)
            
            # Add data rows
            for horizon in time_horizons:
                horizon_results = self.evaluation_results['by_time_horizon'].get(horizon, {})
                
                row = f"| {horizon} min |"
                
                for metric in available_metrics:
                    if metric in horizon_results:
                        row += f" {horizon_results[metric]:.2f} |"
                    else:
                        row += " - |"
                
                report.append(row)
            
            report.append("")
        
        # Add target class analysis if available
        if self.evaluation_results['by_target_class']:
            report.append("## Performance by Target Class")
            
            # Create table header
            header = "| Target Class |"
            separator = "|-------------|"
            
            metrics = ['mae', 'rmse', 'cep50', 'accuracy_100m']
            available_metrics = []
            
            for metric in metrics:
                for target_class, class_results in self.evaluation_results['by_target_class'].items():
                    if metric in class_results:
                        if metric not in available_metrics:
                            available_metrics.append(metric)
            
            for metric in available_metrics:
                header += f" {metric.upper()} |"
                separator += "--------|"
            
            report.append(header)
            report.append(separator)
            
            # Add data rows
            for target_class, class_results in self.evaluation_results['by_target_class'].items():
                row = f"| {target_class} |"
                
                for metric in available_metrics:
                    if metric in class_results:
                        row += f" {class_results[metric]:.2f} |"
                    else:
                        row += " - |"
                
                report.append(row)
            
            report.append("")
        
        # Add confidence region evaluation if available
        if self.evaluation_results['confidence_regions']:
            report.append("## Confidence Region Evaluation")
            
            # Create table
            report.append("| Confidence Level | Accuracy | Average Area (km²) |")
            report.append("|-----------------|----------|-------------------|")
            
            for level, metrics in sorted(self.evaluation_results['confidence_regions'].items()):
                if metrics['count'] > 0:
                    report.append(f"| {level*100:.0f}% | {metrics['accuracy']*100:.1f}% | {metrics['avg_area']:.2f} |")
            
            report.append("")
        
        # Add conclusions and recommendations
        report.append("## Conclusions and Recommendations")
        
        # Best method
        if methods and len(methods) > 1:
            scores = self.calculate_performance_score()
            if scores:
                best_method = max(scores.items(), key=lambda x: x[1])[0]
                report.append(f"- The best performing prediction method is **{best_method}**.")
        
        # Performance variation with time horizon
        if 'time_degradation' in self.evaluation_results and 'mae' in self.evaluation_results['time_degradation']:
            mae_degradation = self.evaluation_results['time_degradation']['mae']
            if len(mae_degradation) > 1:
                min_horizon = min(mae_degradation.keys())
                max_horizon = max(mae_degradation.keys())
                
                if min_horizon in mae_degradation and max_horizon in mae_degradation:
                    degradation_factor = mae_degradation[max_horizon] / mae_degradation[min_horizon]
                    report.append(f"- Prediction error increases by a factor of {degradation_factor:.1f}x from {min_horizon} to {max_horizon} minutes.")
        
        # Confidence region reliability
        if self.evaluation_results['confidence_regions']:
            for level, metrics in sorted(self.evaluation_results['confidence_regions'].items()):
                if metrics['count'] > 0:
                    if abs(metrics['accuracy'] - level) > 0.1:
                        if metrics['accuracy'] < level:
                            report.append(f"- {level*100:.0f}% confidence regions are overconfident (actual: {metrics['accuracy']*100:.1f}%).")
                        else:
                            report.append(f"- {level*100:.0f}% confidence regions are underconfident (actual: {metrics['accuracy']*100:.1f}%).")
        
        # Practical recommendation
        report.append("- For operational use, predictions should be updated at least every 15 minutes to maintain accuracy.")
        
        # Join report lines
        report_text = "\n".join(report)
        
        # Save report if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            
            if self.verbose:
                print(f"Evaluation report saved to {output_file}")
        
        return report_text
    
    def _calculate_prediction_errors(self, predictions=None):
        """
        Calculate errors between predictions and ground truth.
        
        Args:
            predictions (dict): Dictionary of predictions (optional)
            
        Returns:
            list: List of error details
        """
        # Use all predictions if not specified
        if predictions is None:
            predictions = self.predictions
        
        # Initialize error list
        errors = []
        
        # Process each prediction
        for pred_key, prediction in predictions.items():
            if not isinstance(prediction, dict):
                continue
            
            # Get target ID and time horizon
            target_id = prediction.get('target_id')
            time_horizon = prediction.get('minutes_ahead')
            method = prediction.get('method')
            
            if target_id is None or time_horizon is None:
                continue
            
            # Get prediction time
            prediction_time = self._get_prediction_time(prediction)
            if prediction_time is None:
                continue
            
            # Get ground truth position at prediction time
            ground_truth = self._get_ground_truth(target_id, prediction_time)
            if ground_truth is None:
                continue
            
            # Skip stationary targets if configured
            if self.config['exclude_stationary'] and 'speed' in ground_truth:
                if ground_truth['speed'] < self.config['min_speed_threshold']:
                    continue
            
            # Calculate error distance
            error_distance = self._calculate_distance(
                (prediction.get('pred_longitude', prediction.get('longitude')),
                 prediction.get('pred_latitude', prediction.get('latitude'))),
                (ground_truth['longitude'], ground_truth['latitude'])
            )
            
            # Create error entry
            error_entry = {
                'prediction_key': pred_key,
                'target_id': target_id,
                'time_horizon': time_horizon,
                'method': method,
                'ground_truth': ground_truth,
                'prediction': {
                    'longitude': prediction.get('pred_longitude', prediction.get('longitude')),
                    'latitude': prediction.get('pred_latitude', prediction.get('latitude'))
                },
                'error_distance': error_distance
            }
            
            # Add terrain type if available
            if 'land_use_type' in ground_truth:
                error_entry['terrain_type'] = ground_truth['land_use_type']
            
            # Add path error if path prediction
            if 'path' in prediction:
                path_error = self._calculate_path_error(prediction['path'], ground_truth)
                if path_error is not None:
                    error_entry['path_error'] = path_error
            
            errors.append(error_entry)
        
        return errors
    
    def _get_prediction_time(self, prediction):
        """
        Get the time for which the prediction was made.
        
        Args:
            prediction (dict): Prediction data
            
        Returns:
            datetime.datetime: Prediction time
        """
        if 'prediction_time' in prediction:
            # Use explicit prediction time if available
            return prediction['prediction_time']
        
        if 'reference_time' in prediction and 'minutes_ahead' in prediction:
            # Calculate from reference time and horizon
            ref_time = prediction['reference_time']
            minutes = prediction['minutes_ahead']
            
            if isinstance(ref_time, str):
                try:
                    # Parse datetime string
                    ref_time = datetime.datetime.fromisoformat(ref_time.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        # Try alternative format
                        ref_time = datetime.datetime.strptime(ref_time, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        return None
            
            # Add time delta
            return ref_time + datetime.timedelta(minutes=minutes)
        
        return None
    
    def _get_ground_truth(self, target_id, prediction_time):
        """
        Get ground truth position for a target at a specific time.
        
        Args:
            target_id: Target ID
            prediction_time (datetime.datetime): Time for which to get ground truth
            
        Returns:
            dict: Ground truth data
        """
        if target_id is None or prediction_time is None:
            return None
        
        # Filter target observations by ID
        target_data = self.targets_df[self.targets_df['target_id'] == target_id].copy()
        
        if target_data.empty:
            return None
        
        # Ensure timestamp column is datetime
        if 'timestamp' in target_data.columns:
            if target_data['timestamp'].dtype == 'object':
                try:
                    target_data['timestamp'] = pd.to_datetime(target_data['timestamp'])
                except:
                    if self.verbose:
                        print(f"Could not convert timestamps for target {target_id}")
                    return None
        else:
            return None
        
        # Find closest observations before and after prediction time
        before_obs = target_data[target_data['timestamp'] <= prediction_time].sort_values('timestamp')
        after_obs = target_data[target_data['timestamp'] >= prediction_time].sort_values('timestamp')
        
        if before_obs.empty and after_obs.empty:
            return None
        
        if before_obs.empty:
            # If no observations before prediction time, use first observation after
            return after_obs.iloc[0].to_dict()
        
        if after_obs.empty:
            # If no observations after prediction time, use last observation before
            return before_obs.iloc[-1].to_dict()
        
        # Get closest observations before and after
        before = before_obs.iloc[-1]
        after = after_obs.iloc[0]
        
        # If exact match exists, use it
        if before['timestamp'] == prediction_time:
            return before.to_dict()
        
        if after['timestamp'] == prediction_time:
            return after.to_dict()
        
        # Interpolate between closest observations
        time_diff = (after['timestamp'] - before['timestamp']).total_seconds()
        
        if time_diff <= 0:
            return before.to_dict()
        
        # Calculate time ratio
        pred_diff = (prediction_time - before['timestamp']).total_seconds()
        ratio = pred_diff / time_diff
        
        # Interpolate position
        interpolated = {
            'target_id': target_id,
            'timestamp': prediction_time,
            'longitude': before['longitude'] + ratio * (after['longitude'] - before['longitude']),
            'latitude': before['latitude'] + ratio * (after['latitude'] - before['latitude'])
        }
        
        # Interpolate additional fields if available
        for field in ['elevation', 'speed', 'heading']:
            if field in before and field in after:
                interpolated[field] = before[field] + ratio * (after[field] - before[field])
        
        # Copy non-interpolated fields
        for field in before.index:
            if field not in interpolated and field not in ['longitude', 'latitude', 'elevation', 'speed', 'heading']:
                interpolated[field] = before[field]
        
        return interpolated
    
    def _calculate_distance(self, point1, point2):
        """
        Calculate distance between two geographic points.
        
        Args:
            point1 (tuple): First point (longitude, latitude)
            point2 (tuple): Second point (longitude, latitude)
            
        Returns:
            float: Distance in meters
        """
        if point1 is None or point2 is None:
            return float('inf')
        
        # Extract coordinates
        lon1, lat1 = point1
        lon2, lat2 = point2
        
        if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
            return float('inf')
        
        # If meters_per_degree is provided in config, use it for conversion
        if self.config['meters_per_degree'] is not None:
            # Simple Euclidean distance
            dx = (lon2 - lon1) * self.config['meters_per_degree']
            dy = (lat2 - lat1) * self.config['meters_per_degree']
            return math.sqrt(dx**2 + dy**2)
        
        # Otherwise, use Haversine formula for accurate distance
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        
        return c * r
    
    def _calculate_path_error(self, path, ground_truth):
        """
        Calculate error between a predicted path and ground truth position.
        
        Uses minimum distance from ground truth to any point on path.
        
        Args:
            path (list): List of points along predicted path
            ground_truth (dict): Ground truth position
            
        Returns:
            float: Path error distance in meters
        """
        if not path or ground_truth is None:
            return None
        
        # Extract ground truth coordinates
        gt_point = (ground_truth['longitude'], ground_truth['latitude'])
        
        # Calculate distances to all points on path
        distances = []
        
        for path_point in path:
            if isinstance(path_point, (list, tuple)) and len(path_point) >= 2:
                # Direct coordinates
                distances.append(self._calculate_distance(path_point, gt_point))
            elif isinstance(path_point, dict) and 'longitude' in path_point and 'latitude' in path_point:
                # Dictionary with coordinates
                point = (path_point['longitude'], path_point['latitude'])
                distances.append(self._calculate_distance(point, gt_point))
        
        if not distances:
            return None
        
        # Return minimum distance
        return min(distances)
    
    def _get_unique_methods(self):
        """
        Get list of unique prediction methods.
        
        Returns:
            list: List of method names
        """
        methods = []
        
        for pred in self.predictions.values():
            if isinstance(pred, dict) and 'method' in pred:
                method = pred['method']
                if method not in methods:
                    methods.append(method)
        
        return methods
    
    def _point_in_region(self, point, region):
        """
        Check if a point is inside a confidence region.
        
        Args:
            point (tuple): Point coordinates (longitude, latitude)
            region (dict): Confidence region data
            
        Returns:
            bool: Whether point is inside region
        """
        if not point or not region:
            return False
        
        # Check if region has explicit polygon or points
        if 'polygon' in region:
            polygon = region['polygon']
            
            if isinstance(polygon, list) and len(polygon) >= 3:
                # Create Shapely polygon
                from shapely.geometry import Polygon, Point as ShapelyPoint # type: ignore
                
                try:
                    poly = Polygon(polygon)
                    point_obj = ShapelyPoint(point)
                    
                    return poly.contains(point_obj)
                except:
                    pass
        
        elif 'points' in region:
            points = region['points']
            
            if isinstance(points, list) and len(points) >= 3:
                # Create Shapely polygon from convex hull
                from shapely.geometry import Polygon, Point as ShapelyPoint # type: ignore
                from scipy.spatial import ConvexHull
                
                try:
                    hull = ConvexHull(points)
                    hull_points = [points[i] for i in hull.vertices]
                    
                    poly = Polygon(hull_points)
                    point_obj = ShapelyPoint(point)
                    
                    return poly.contains(point_obj)
                except:
                    pass
        
        # If no explicit region representation or previous checks failed,
        # check if region has center and radius
        if 'center' in region and 'radius' in region:
            center = region['center']
            radius = region['radius']
            
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                center_point = (center[0], center[1])
                distance = self._calculate_distance(point, center_point)
                
                return distance <= radius
        
        return False