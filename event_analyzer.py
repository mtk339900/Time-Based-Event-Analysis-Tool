#!/usr/bin/env python3
"""
Time-Based Event Analysis Tool
A comprehensive tool for analyzing temporal patterns in event datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sqlite3
import warnings
from datetime import datetime, timezone
from dateutil import parser as date_parser
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class AnalysisConfig:
    """Configuration class for analysis parameters."""
    time_interval: str = 'hour'  # minute, hour, day, week, month
    timezone: str = 'UTC'
    anomaly_threshold: float = 2.0  # Standard deviations for anomaly detection
    min_samples_for_anomaly: int = 10
    pattern_detection_window: int = 7  # Days for pattern detection


class DataLoader:
    """Handles loading data from various sources."""
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} records from CSV: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_json(file_path: str) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.json_normalize(data)
            
            logger.info(f"Successfully loaded {len(df)} records from JSON: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_database(connection_string: str, query: str) -> pd.DataFrame:
        """Load data from database connection."""
        try:
            if connection_string.startswith('sqlite'):
                db_path = connection_string.replace('sqlite:///', '').replace('sqlite://', '')
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
            else:
                # For other databases, would need appropriate drivers
                raise NotImplementedError("Only SQLite connections are implemented in this version")
            
            logger.info(f"Successfully loaded {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            raise


class TimestampProcessor:
    """Handles timestamp parsing and normalization."""
    
    def __init__(self, target_timezone: str = 'UTC'):
        self.target_timezone = pytz.timezone(target_timezone)
    
    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        try:
            # Try to parse with dateutil parser (handles most formats)
            dt = date_parser.parse(timestamp_str)
            
            # If no timezone info, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            
            # Convert to target timezone
            return dt.astimezone(self.target_timezone)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            return None
    
    def normalize_timestamps(self, df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Normalize timestamps in the dataframe."""
        logger.info(f"Normalizing timestamps in column: {timestamp_column}")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Parse timestamps
        df['parsed_timestamp'] = df[timestamp_column].apply(self.parse_timestamp)
        
        # Remove rows with unparseable timestamps
        initial_count = len(df)
        df = df.dropna(subset=['parsed_timestamp'])
        final_count = len(df)
        
        if initial_count != final_count:
            logger.warning(f"Dropped {initial_count - final_count} rows with invalid timestamps")
        
        # Set as datetime index
        df = df.set_index('parsed_timestamp').sort_index()
        
        return df


class EventAnalyzer:
    """Core analysis engine for temporal event patterns."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def group_by_time_interval(self, df: pd.DataFrame, 
                             interval: str = None) -> pd.DataFrame:
        """Group events by specified time intervals."""
        if interval is None:
            interval = self.config.time_interval
        
        # Map interval names to pandas frequency strings
        interval_map = {
            'minute': 'T',
            'hour': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }
        
        freq = interval_map.get(interval, 'H')
        
        # Group by time interval and count events
        grouped = df.groupby(pd.Grouper(freq=freq)).size()
        grouped.name = 'event_count'
        
        return grouped.to_frame().reset_index()
    
    def calculate_frequency_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate event frequency distributions."""
        results = {}
        
        # Overall statistics
        event_counts = self.group_by_time_interval(df)
        results['overall_stats'] = {
            'total_events': len(df),
            'total_periods': len(event_counts),
            'mean_events_per_period': event_counts['event_count'].mean(),
            'std_events_per_period': event_counts['event_count'].std(),
            'min_events_per_period': event_counts['event_count'].min(),
            'max_events_per_period': event_counts['event_count'].max()
        }
        
        # Hourly distribution
        df_copy = df.reset_index()
        df_copy['hour'] = df_copy['parsed_timestamp'].dt.hour
        hourly_dist = df_copy.groupby('hour').size()
        results['hourly_distribution'] = hourly_dist.to_dict()
        
        # Daily distribution
        df_copy['day_of_week'] = df_copy['parsed_timestamp'].dt.day_name()
        daily_dist = df_copy.groupby('day_of_week').size()
        results['daily_distribution'] = daily_dist.to_dict()
        
        # Monthly distribution
        df_copy['month'] = df_copy['parsed_timestamp'].dt.month_name()
        monthly_dist = df_copy.groupby('month').size()
        results['monthly_distribution'] = monthly_dist.to_dict()
        
        return results
    
    def identify_peak_periods(self, df: pd.DataFrame, 
                            top_n: int = 10) -> pd.DataFrame:
        """Identify peak activity periods."""
        grouped = self.group_by_time_interval(df)
        
        # Sort by event count and get top N
        peak_periods = grouped.nlargest(top_n, 'event_count')
        
        return peak_periods
    
    def identify_idle_periods(self, df: pd.DataFrame, 
                            bottom_n: int = 10) -> pd.DataFrame:
        """Identify idle/low activity periods."""
        grouped = self.group_by_time_interval(df)
        
        # Filter out zero counts and get bottom N
        non_zero = grouped[grouped['event_count'] > 0]
        idle_periods = non_zero.nsmallest(bottom_n, 'event_count')
        
        return idle_periods
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in event patterns."""
        grouped = self.group_by_time_interval(df)
        
        if len(grouped) < self.config.min_samples_for_anomaly:
            return {'anomalies': [], 'message': 'Insufficient data for anomaly detection'}
        
        # Calculate z-scores
        counts = grouped['event_count'].values
        z_scores = np.abs(stats.zscore(counts))
        
        # Identify anomalies
        anomaly_threshold = self.config.anomaly_threshold
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
        
        anomalies = []
        for idx in anomaly_indices:
            anomalies.append({
                'timestamp': grouped.iloc[idx]['parsed_timestamp'],
                'event_count': grouped.iloc[idx]['event_count'],
                'z_score': z_scores[idx],
                'type': 'spike' if counts[idx] > np.mean(counts) else 'drop'
            })
        
        return {
            'anomalies': anomalies,
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(grouped) * 100
        }
    
    def detect_recurring_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect recurring time patterns."""
        df_copy = df.reset_index()
        
        patterns = {}
        
        # Daily patterns (by hour)
        hourly_pattern = df_copy.groupby(df_copy['parsed_timestamp'].dt.hour).size()
        patterns['hourly_pattern'] = {
            'data': hourly_pattern.to_dict(),
            'peak_hour': hourly_pattern.idxmax(),
            'peak_count': hourly_pattern.max(),
            'variation_coefficient': hourly_pattern.std() / hourly_pattern.mean()
        }
        
        # Weekly patterns (by day of week)
        weekly_pattern = df_copy.groupby(df_copy['parsed_timestamp'].dt.dayofweek).size()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns['weekly_pattern'] = {
            'data': {day_names[i]: weekly_pattern.get(i, 0) for i in range(7)},
            'peak_day': day_names[weekly_pattern.idxmax()],
            'peak_count': weekly_pattern.max(),
            'variation_coefficient': weekly_pattern.std() / weekly_pattern.mean()
        }
        
        # Monthly patterns (by day of month)
        if len(df_copy) > 30:  # Only if we have enough data
            monthly_pattern = df_copy.groupby(df_copy['parsed_timestamp'].dt.day).size()
            patterns['monthly_pattern'] = {
                'data': monthly_pattern.to_dict(),
                'peak_day': monthly_pattern.idxmax(),
                'peak_count': monthly_pattern.max(),
                'variation_coefficient': monthly_pattern.std() / monthly_pattern.mean()
            }
        
        return patterns


class DataFilter:
    """Handles filtering operations on event data."""
    
    @staticmethod
    def filter_by_column(df: pd.DataFrame, column: str, 
                        values: Union[str, List[str]]) -> pd.DataFrame:
        """Filter dataframe by column values."""
        if isinstance(values, str):
            values = [values]
        
        filtered_df = df[df[column].isin(values)]
        logger.info(f"Filtered data: {len(filtered_df)} records remain after filtering {column}")
        
        return filtered_df
    
    @staticmethod
    def filter_by_time_range(df: pd.DataFrame, start_time: str = None, 
                           end_time: str = None) -> pd.DataFrame:
        """Filter dataframe by time range."""
        filtered_df = df.copy()
        
        if start_time:
            start_dt = date_parser.parse(start_time)
            filtered_df = filtered_df[filtered_df.index >= start_dt]
        
        if end_time:
            end_dt = date_parser.parse(end_time)
            filtered_df = filtered_df[filtered_df.index <= end_dt]
        
        logger.info(f"Time range filter: {len(filtered_df)} records remain")
        
        return filtered_df


class Visualizer:
    """Handles visualization of analysis results."""
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
    
    def plot_time_series(self, df: pd.DataFrame, title: str = "Event Count Over Time",
                        save_path: str = None) -> plt.Figure:
        """Create time series line plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grouped = df.groupby(pd.Grouper(freq='H')).size()
        grouped.plot(ax=ax, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Event Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to: {save_path}")
        
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame, title: str = "Activity Heatmap",
                    save_path: str = None) -> plt.Figure:
        """Create heatmap showing activity patterns."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create hour vs day of week heatmap
        df_copy = df.reset_index()
        df_copy['hour'] = df_copy['parsed_timestamp'].dt.hour
        df_copy['day_name'] = df_copy['parsed_timestamp'].dt.day_name()
        
        heatmap_data = df_copy.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Event Count'}, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to: {save_path}")
        
        return fig
    
    def plot_histogram(self, df: pd.DataFrame, column: str = 'event_count',
                      title: str = "Event Count Distribution", 
                      save_path: str = None) -> plt.Figure:
        """Create histogram of event counts."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if column == 'event_count':
            data = df.groupby(pd.Grouper(freq='H')).size()
        else:
            data = df[column]
        
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Histogram saved to: {save_path}")
        
        return fig
    
    def plot_anomalies(self, df: pd.DataFrame, anomalies: List[Dict],
                      title: str = "Anomaly Detection", 
                      save_path: str = None) -> plt.Figure:
        """Plot time series with anomalies highlighted."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grouped = df.groupby(pd.Grouper(freq='H')).size()
        grouped.plot(ax=ax, linewidth=2, label='Event Count')
        
        # Highlight anomalies
        for anomaly in anomalies:
            timestamp = pd.to_datetime(anomaly['timestamp'])
            count = anomaly['event_count']
            color = 'red' if anomaly['type'] == 'spike' else 'blue'
            ax.scatter(timestamp, count, color=color, s=100, alpha=0.7,
                      label=f"{anomaly['type'].title()} Anomaly")
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Event Count', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly plot saved to: {save_path}")
        
        return fig


class ResultExporter:
    """Handles exporting results to various formats."""
    
    @staticmethod
    def export_to_csv(data: Union[pd.DataFrame, Dict], file_path: str):
        """Export data to CSV file."""
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=True)
            else:
                # Convert dict to DataFrame
                df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                df.to_csv(file_path, index=False)
            
            logger.info(f"Data exported to CSV: {file_path}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    @staticmethod
    def export_to_json(data: Union[Dict, pd.DataFrame], file_path: str):
        """Export data to JSON file."""
        try:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to JSON
                json_data = data.to_dict(orient='records')
            else:
                json_data = data
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"Data exported to JSON: {file_path}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise


class EventAnalysisTool:
    """Main class orchestrating the entire analysis workflow."""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.data_loader = DataLoader()
        self.timestamp_processor = TimestampProcessor(self.config.timezone)
        self.analyzer = EventAnalyzer(self.config)
        self.filter = DataFilter()
        self.visualizer = Visualizer()
        self.exporter = ResultExporter()
        
        self.raw_data = None
        self.processed_data = None
        self.analysis_results = {}
    
    def load_data(self, source: str, source_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Load data from specified source."""
        if source_type == 'csv':
            self.raw_data = self.data_loader.load_csv(source, **kwargs)
        elif source_type == 'json':
            self.raw_data = self.data_loader.load_json(source)
        elif source_type == 'database':
            query = kwargs.get('query', 'SELECT * FROM events')
            self.raw_data = self.data_loader.load_database(source, query)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return self.raw_data
    
    def preprocess_data(self, timestamp_column: str, 
                       filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Preprocess the loaded data."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Normalize timestamps
        self.processed_data = self.timestamp_processor.normalize_timestamps(
            self.raw_data, timestamp_column
        )
        
        # Apply filters if provided
        if filters:
            for filter_type, filter_params in filters.items():
                if filter_type == 'column':
                    for column, values in filter_params.items():
                        self.processed_data = self.filter.filter_by_column(
                            self.processed_data, column, values
                        )
                elif filter_type == 'time_range':
                    self.processed_data = self.filter.filter_by_time_range(
                        self.processed_data, **filter_params
                    )
        
        logger.info(f"Data preprocessing completed. {len(self.processed_data)} records ready for analysis.")
        return self.processed_data
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete temporal analysis."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        logger.info("Starting comprehensive temporal analysis...")
        
        # Frequency distribution analysis
        self.analysis_results['frequency_distribution'] = \
            self.analyzer.calculate_frequency_distribution(self.processed_data)
        
        # Peak and idle period identification
        self.analysis_results['peak_periods'] = \
            self.analyzer.identify_peak_periods(self.processed_data)
        
        self.analysis_results['idle_periods'] = \
            self.analyzer.identify_idle_periods(self.processed_data)
        
        # Anomaly detection
        self.analysis_results['anomalies'] = \
            self.analyzer.detect_anomalies(self.processed_data)
        
        # Pattern detection
        self.analysis_results['recurring_patterns'] = \
            self.analyzer.detect_recurring_patterns(self.processed_data)
        
        logger.info("Analysis completed successfully.")
        return self.analysis_results
    
    def generate_visualizations(self, output_dir: str = "output"):
        """Generate all visualizations."""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Time series plot
        self.visualizer.plot_time_series(
            self.processed_data,
            title="Event Timeline Analysis",
            save_path=f"{output_dir}/time_series.png"
        )
        
        # Activity heatmap
        self.visualizer.plot_heatmap(
            self.processed_data,
            title="Weekly Activity Heatmap",
            save_path=f"{output_dir}/activity_heatmap.png"
        )
        
        # Event count histogram
        self.visualizer.plot_histogram(
            self.processed_data,
            title="Event Count Distribution",
            save_path=f"{output_dir}/event_histogram.png"
        )
        
        # Anomaly plot if anomalies were detected
        if 'anomalies' in self.analysis_results and self.analysis_results['anomalies']['anomalies']:
            self.visualizer.plot_anomalies(
                self.processed_data,
                self.analysis_results['anomalies']['anomalies'],
                title="Detected Anomalies",
                save_path=f"{output_dir}/anomalies.png"
            )
        
        logger.info(f"All visualizations saved to: {output_dir}")
    
    def export_results(self, output_dir: str = "output"):
        """Export all analysis results."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analysis first.")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export analysis results to JSON
        self.exporter.export_to_json(
            self.analysis_results,
            f"{output_dir}/analysis_results.json"
        )
        
        # Export processed data to CSV
        self.exporter.export_to_csv(
            self.processed_data,
            f"{output_dir}/processed_data.csv"
        )
        
        # Export peak periods to CSV
        if isinstance(self.analysis_results['peak_periods'], pd.DataFrame):
            self.exporter.export_to_csv(
                self.analysis_results['peak_periods'],
                f"{output_dir}/peak_periods.csv"
            )
        
        # Export idle periods to CSV
        if isinstance(self.analysis_results['idle_periods'], pd.DataFrame):
            self.exporter.export_to_csv(
                self.analysis_results['idle_periods'],
                f"{output_dir}/idle_periods.csv"
            )
        
        logger.info(f"All results exported to: {output_dir}")
    
    def print_summary(self):
        """Print a comprehensive analysis summary."""
        if not self.analysis_results:
            print("No analysis results available.")
            return
        
        print("=" * 80)
        print("EVENT ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        freq_dist = self.analysis_results.get('frequency_distribution', {})
        overall_stats = freq_dist.get('overall_stats', {})
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Events: {overall_stats.get('total_events', 'N/A'):,}")
        print(f"  Analysis Periods: {overall_stats.get('total_periods', 'N/A'):,}")
        print(f"  Mean Events/Period: {overall_stats.get('mean_events_per_period', 0):.2f}")
        print(f"  Std Dev: {overall_stats.get('std_events_per_period', 0):.2f}")
        
        # Peak periods
        print(f"\nPEAK ACTIVITY PERIODS:")
        peak_periods = self.analysis_results.get('peak_periods')
        if isinstance(peak_periods, pd.DataFrame) and not peak_periods.empty:
            for i, row in peak_periods.head(3).iterrows():
                print(f"  {row['parsed_timestamp']}: {row['event_count']} events")
        
        # Anomalies
        print(f"\nANOMALY DETECTION:")
        anomalies = self.analysis_results.get('anomalies', {})
        print(f"  Total Anomalies: {anomalies.get('total_anomalies', 0)}")
        print(f"  Anomaly Rate: {anomalies.get('anomaly_rate', 0):.2f}%")
        
        # Patterns
        print(f"\nRECURRING PATTERNS:")
        patterns = self.analysis_results.get('recurring_patterns', {})
        if 'hourly_pattern' in patterns:
            hourly = patterns['hourly_pattern']
            print(f"  Peak Hour: {hourly.get('peak_hour', 'N/A')}:00 ({hourly.get('peak_count', 0)} events)")
        
        if 'weekly_pattern' in patterns:
            weekly = patterns['weekly_pattern']
            print(f"  Peak Day: {weekly.get('peak_day', 'N/A')} ({weekly.get('peak_count', 0)} events)")
        
        print("=" * 80)


def main():
    """Example usage of the Event Analysis Tool."""
    # Configuration
    config = AnalysisConfig(
        time_interval='hour',
        timezone='UTC',
        anomaly_threshold=2.0
    )
    
    # Initialize the tool
    tool = EventAnalysisTool(config)
    
    try:
        # Example: Create sample data for demonstration
        # In real usage, replace this with actual data loading
        print("Creating sample event data for demonstration...")
        
        # Generate sample data
        dates = pd.date_range('2024-01-01', periods=1000, freq='3H')
        np.random.seed(42)
        
        # Create realistic patterns with some noise and anomalies
        base_counts = np.random.poisson(50, len(dates))
        
        # Add hourly patterns (higher activity during business hours)
        hourly_multiplier = np.array([0.5 + 0.5 * np.sin((d.hour - 12) * np.pi / 12) for d in dates])
        
        # Add weekly patterns (lower activity on weekends)
        weekly_multiplier = np.array([0.7 if d.weekday() >= 5 else 1.0 for d in dates])
        
        # Combine patterns
        event_counts = (base_counts * hourly_multiplier * weekly_multiplier).astype(int)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(len(dates), 10, replace=False)
        event_counts[anomaly_indices] *= 3  # Create spikes
        
        # Create sample events
        sample_events = []
        event_types = ['login', 'transaction', 'error', 'warning', 'info']
        user_ids = [f'user_{i:03d}' for i in range(1, 101)]
        
        for i, (timestamp, count) in enumerate(zip(dates, event_counts)):
            for _ in range(count):
                sample_events.append({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_type': np.random.choice(event_types),
                    'user_id': np.random.choice(user_ids),
                    'category': np.random.choice(['system', 'user_action', 'error'])
                })
        
        # Create sample CSV file
        sample_df = pd.DataFrame(sample_events)
        sample_df.to_csv('sample_events.csv', index=False)
        print(f"Generated {len(sample_events)} sample events")
        
        # Load the sample data
        print("\n1. Loading data from CSV...")
        tool.load_data('sample_events.csv', source_type='csv')
        
        # Preprocess data with optional filters
        print("2. Preprocessing data...")
        filters = {
            'column': {
                'event_type': ['login', 'transaction', 'error']  # Filter specific event types
            }
        }
        tool.preprocess_data('timestamp', filters=filters)
        
        # Run comprehensive analysis
        print("3. Running temporal analysis...")
        results = tool.run_analysis()
        
        # Generate visualizations
        print("4. Generating visualizations...")
        tool.generate_visualizations('analysis_output')
        
        # Export results
        print("5. Exporting results...")
        tool.export_results('analysis_output')
        
        # Display summary
        print("6. Analysis Summary:")
        tool.print_summary()
        
        print(f"\nAnalysis complete! Check the 'analysis_output' directory for:")
        print("  - Visualizations (PNG files)")
        print("  - Analysis results (JSON)")
        print("  - Processed data (CSV)")
        print("  - Peak/Idle periods (CSV)")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()


# Additional utility functions for advanced analysis

class AdvancedAnalyzer:
    """Advanced analysis methods for complex temporal patterns."""
    
    @staticmethod
    def seasonal_decomposition(df: pd.DataFrame, period: int = 24) -> Dict[str, Any]:
        """Perform seasonal decomposition of time series."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            grouped = df.groupby(pd.Grouper(freq='H')).size()
            
            # Ensure we have enough data points
            if len(grouped) < 2 * period:
                return {'error': 'Insufficient data for seasonal decomposition'}
            
            decomposition = seasonal_decompose(grouped, model='additive', period=period)
            
            return {
                'trend': decomposition.trend.dropna().to_dict(),
                'seasonal': decomposition.seasonal.dropna().to_dict(),
                'residual': decomposition.resid.dropna().to_dict()
            }
        except ImportError:
            return {'error': 'statsmodels not available for seasonal decomposition'}
        except Exception as e:
            return {'error': f'Decomposition failed: {str(e)}'}
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame, 
                           columns: List[str]) -> Dict[str, Any]:
        """Analyze correlations between different event attributes."""
        correlations = {}
        
        for col in columns:
            if col in df.columns:
                # Create time-based aggregation for each category
                grouped = df.groupby([pd.Grouper(freq='H'), col]).size().unstack(fill_value=0)
                corr_matrix = grouped.corr()
                correlations[col] = corr_matrix.to_dict()
        
        return correlations
    
    @staticmethod
    def burst_detection(df: pd.DataFrame, window_size: int = 60) -> List[Dict]:
        """Detect burst periods in event streams."""
        grouped = df.groupby(pd.Grouper(freq='T')).size()  # Minute-level grouping
        
        # Calculate rolling statistics
        rolling_mean = grouped.rolling(window=window_size).mean()
        rolling_std = grouped.rolling(window=window_size).std()
        
        # Detect bursts (values significantly above rolling average)
        threshold = rolling_mean + 2 * rolling_std
        burst_mask = grouped > threshold
        
        bursts = []
        in_burst = False
        burst_start = None
        
        for timestamp, is_burst in burst_mask.items():
            if is_burst and not in_burst:
                # Start of burst
                burst_start = timestamp
                in_burst = True
            elif not is_burst and in_burst:
                # End of burst
                bursts.append({
                    'start': burst_start,
                    'end': timestamp,
                    'duration_minutes': (timestamp - burst_start).total_seconds() / 60,
                    'peak_rate': grouped[burst_start:timestamp].max()
                })
                in_burst = False
        
        return bursts


class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, analysis_tool: EventAnalysisTool):
        self.tool = analysis_tool
    
    def generate_html_report(self, output_path: str = "analysis_report.html"):
        """Generate comprehensive HTML report."""
        if not self.tool.analysis_results:
            raise ValueError("No analysis results available")
        
        html_content = self._create_html_template()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
    
    def _create_html_template(self) -> str:
        """Create HTML report template with analysis results."""
        results = self.tool.analysis_results
        freq_dist = results.get('frequency_distribution', {})
        overall_stats = freq_dist.get('overall_stats', {})
        anomalies = results.get('anomalies', {})
        patterns = results.get('recurring_patterns', {})
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Event Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .anomaly {{ background: #ffe8e8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Event Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="metric">
                    <strong>Total Events:</strong> {overall_stats.get('total_events', 'N/A'):,}
                </div>
                <div class="metric">
                    <strong>Analysis Periods:</strong> {overall_stats.get('total_periods', 'N/A'):,}
                </div>
                <div class="metric">
                    <strong>Mean Events per Period:</strong> {overall_stats.get('mean_events_per_period', 0):.2f}
                </div>
                <div class="metric">
                    <strong>Standard Deviation:</strong> {overall_stats.get('std_events_per_period', 0):.2f}
                </div>
            </div>
            
            <div class="section">
                <h2>Anomaly Detection</h2>
                <div class="metric">
                    <strong>Total Anomalies:</strong> {anomalies.get('total_anomalies', 0)}
                </div>
                <div class="metric">
                    <strong>Anomaly Rate:</strong> {anomalies.get('anomaly_rate', 0):.2f}%
                </div>
            </div>
            
            <div class="section">
                <h2>Recurring Patterns</h2>
                {self._format_patterns_html(patterns)}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>The following visualizations were generated:</p>
                <ul>
                    <li>Time Series Plot (time_series.png)</li>
                    <li>Activity Heatmap (activity_heatmap.png)</li>
                    <li>Event Histogram (event_histogram.png)</li>
                    <li>Anomaly Detection Plot (anomalies.png)</li>
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        return html
    
    def _format_patterns_html(self, patterns: Dict) -> str:
        """Format pattern analysis results for HTML."""
        html_parts = []
        
        if 'hourly_pattern' in patterns:
            hourly = patterns['hourly_pattern']
            html_parts.append(f"""
                <div class="metric">
                    <strong>Hourly Pattern:</strong><br>
                    Peak Hour: {hourly.get('peak_hour', 'N/A')}:00 
                    ({hourly.get('peak_count', 0)} events)<br>
                    Variation Coefficient: {hourly.get('variation_coefficient', 0):.3f}
                </div>
            """)
        
        if 'weekly_pattern' in patterns:
            weekly = patterns['weekly_pattern']
            html_parts.append(f"""
                <div class="metric">
                    <strong>Weekly Pattern:</strong><br>
                    Peak Day: {weekly.get('peak_day', 'N/A')} 
                    ({weekly.get('peak_count', 0)} events)<br>
                    Variation Coefficient: {weekly.get('variation_coefficient', 0):.3f}
                </div>
            """)
        
        return ''.join(html_parts)


# Example configuration templates
class ConfigTemplates:
    """Predefined configuration templates for common use cases."""
    
    @staticmethod
    def get_system_logs_config() -> AnalysisConfig:
        """Configuration optimized for system log analysis."""
        return AnalysisConfig(
            time_interval='minute',
            timezone='UTC',
            anomaly_threshold=3.0,
            min_samples_for_anomaly=30,
            pattern_detection_window=7
        )
    
    @staticmethod
    def get_transaction_config() -> AnalysisConfig:
        """Configuration optimized for transaction analysis."""
        return AnalysisConfig(
            time_interval='hour',
            timezone='UTC',
            anomaly_threshold=2.5,
            min_samples_for_anomaly=24,
            pattern_detection_window=30
        )
    
    @staticmethod
    def get_user_activity_config() -> AnalysisConfig:
        """Configuration optimized for user activity analysis."""
        return AnalysisConfig(
            time_interval='hour',
            timezone='UTC',
            anomaly_threshold=2.0,
            min_samples_for_anomaly=48,
            pattern_detection_window=14
        )
