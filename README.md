# Time-Based Event Analysis Tool

A comprehensive Python tool for analyzing temporal patterns in event datasets such as system logs, transaction records, and activity timelines. This tool provides advanced time-series analysis, anomaly detection, pattern recognition, and visualization capabilities.

## Features

### 🔍 **Data Input Support**
- **CSV files** with flexible parsing options
- **JSON files** (arrays or nested objects)  
- **SQLite databases** (extensible to other databases)
- Robust data validation and error handling

### ⏰ **Timestamp Processing**
- Automatic parsing of various timestamp formats
- Multi-timezone support and normalization
- Intelligent handling of malformed timestamps
- Configurable target timezone conversion

### 📊 **Time-Based Analysis**
- **Configurable time intervals**: minute, hour, day, week, month
- **Peak and idle period identification**
- **Event frequency distributions** over time
- **Recurring pattern detection** (hourly, daily, weekly, monthly patterns)
- **Statistical analysis** with comprehensive metrics

### 🚨 **Anomaly Detection**
- Statistical anomaly detection using z-scores
- Configurable sensitivity thresholds
- Classification of anomalies (spikes vs drops)
- Visual highlighting of detected anomalies

### 🔧 **Data Filtering**
- Filter by event type, user ID, category, or any column
- Time range filtering
- Chainable filter operations
- Memory-efficient filtering operations

### 📈 **Visualizations**
- **Time series line charts** showing event counts over time
- **Activity heatmaps** (hour vs day of week)
- **Event count histograms** for distribution analysis
- **Anomaly detection plots** with highlighted outliers
- High-quality PNG exports (300 DPI)

### 💾 **Export Capabilities**
- **CSV export** for processed data and analysis results
- **JSON export** for structured analysis results
- **PNG export** for all visualizations
- **Comprehensive HTML reports** with embedded analysis

### 🏗️ **Advanced Features**
- Correlation analysis between event attributes
- Burst detection for sudden activity surges
- Seasonal decomposition support
- Modular architecture for easy extension

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies
For seasonal decomposition features:
```bash
pip install statsmodels
```

## Quick Start

### Basic Usage

```python
from event_analyzer import EventAnalysisTool, AnalysisConfig

# Configure analysis parameters
config = AnalysisConfig(
    time_interval='hour',
    timezone='UTC',
    anomaly_threshold=2.0
)

# Initialize the tool
tool = EventAnalysisTool(config)

# Load data from CSV
tool.load_data('your_events.csv')

# Preprocess data (specify timestamp column)
tool.preprocess_data('timestamp')

# Run comprehensive analysis
results = tool.run_analysis()

# Generate visualizations
tool.generate_visualizations('output/')

# Export all results
tool.export_results('output/')

# Print analysis summary
tool.print_summary()
```

### Advanced Usage with Filtering

```python
# Apply filters during preprocessing
filters = {
    'column': {
        'event_type': ['error', 'warning', 'critical'],
        'user_id': ['user_001', 'user_002']
    },
    'time_range': {
        'start_time': '2024-01-01 00:00:00',
        'end_time': '2024-01-31 23:59:59'
    }
}

tool.preprocess_data('timestamp', filters=filters)
```

### Loading from Different Sources

```python
# From JSON file
tool.load_data('events.json', source_type='json')

# From SQLite database
tool.load_data('sqlite:///database.db', 
               source_type='database', 
               query='SELECT * FROM events WHERE date > "2024-01-01"')

# From CSV with custom parameters
tool.load_data('events.csv', 
               source_type='csv', 
               sep=';', 
               encoding='utf-8')
```

## Configuration Options

### AnalysisConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_interval` | str | 'hour' | Grouping interval: 'minute', 'hour', 'day', 'week', 'month' |
| `timezone` | str | 'UTC' | Target timezone for timestamp normalization |
| `anomaly_threshold` | float | 2.0 | Standard deviations for anomaly detection |
| `min_samples_for_anomaly` | int | 10 | Minimum data points required for anomaly detection |
| `pattern_detection_window` | int | 7 | Days window for pattern detection |

### Predefined Configuration Templates

```python
from event_analyzer import ConfigTemplates

# For system logs (high frequency)
config = ConfigTemplates.get_system_logs_config()

# For transaction data (medium frequency)
config = ConfigTemplates.get_transaction_config()

# For user activity (lower frequency)
config = ConfigTemplates.get_user_activity_config()
```

## Data Format Requirements

### CSV Format
Your CSV file should contain at least a timestamp column. Example:
```csv
timestamp,event_type,user_id,category
2024-01-01 10:00:00,login,user_001,authentication
2024-01-01 10:05:00,transaction,user_001,payment
2024-01-01 10:10:00,logout,user_001,authentication
```

### JSON Format
```json
[
  {
    "timestamp": "2024-01-01T10:00:00Z",
    "event_type": "login",
    "user_id": "user_001",
    "category": "authentication"
  },
  {
    "timestamp": "2024-01-01T10:05:00Z",
    "event_type": "transaction",
    "user_id": "user_001",
    "category": "payment"
  }
]
```

## Output Structure

After running the analysis, you'll find the following files in your output directory:

```
output/
├── analysis_results.json      # Complete analysis results
├── processed_data.csv         # Cleaned and processed data
├── peak_periods.csv          # Identified peak activity periods
├── idle_periods.csv          # Identified low activity periods
├── time_series.png           # Time series visualization
├── activity_heatmap.png      # Weekly activity heatmap
├── event_histogram.png       # Event count distribution
├── anomalies.png             # Anomaly detection plot
└── analysis_report.html      # Comprehensive HTML report
```

## Analysis Results

The tool provides comprehensive analysis including:

### Statistical Metrics
- Total event counts
- Mean events per time period
- Standard deviation and variance
- Min/max activity periods

### Pattern Detection
- **Hourly patterns**: Peak hours and activity distribution
- **Daily patterns**: Weekday vs weekend activity
- **Weekly patterns**: Day-of-week variations
- **Monthly patterns**: Seasonal trends (when sufficient data)

### Anomaly Detection
- Statistical outliers using z-score analysis
- Classification of spikes vs drops
- Anomaly timestamps and severity scores
- Overall anomaly rate calculations

### Peak/Idle Analysis
- Top N highest activity periods
- Lowest activity periods (excluding zero counts)
- Activity distribution statistics

## Advanced Features

### Correlation Analysis
```python
from event_analyzer import AdvancedAnalyzer

analyzer = AdvancedAnalyzer()
correlations = analyzer.correlation_analysis(processed_data, ['event_type', 'category'])
```

### Burst Detection
```python
bursts = analyzer.burst_detection(processed_data, window_size=60)
```

### HTML Report Generation
```python
from event_analyzer import ReportGenerator

report_gen = ReportGenerator(tool)
report_gen.generate_html_report('comprehensive_report.html')
```

## Example Use Cases

### System Log Analysis
- Monitor server performance and identify bottlenecks
- Detect unusual error patterns
- Analyze peak usage times for capacity planning

### Transaction Monitoring
- Identify peak trading hours
- Detect fraudulent activity patterns
- Monitor payment system performance

### User Activity Analysis
- Understand user engagement patterns
- Optimize system maintenance windows
- Detect unusual user behavior

### Security Event Analysis
- Monitor login patterns and detect brute force attacks
- Analyze security incident timelines
- Identify recurring security threats

## Performance Considerations

- **Memory Usage**: Tool efficiently handles large datasets using pandas
- **Processing Speed**: Optimized for datasets with millions of events
- **Storage**: Visualizations are saved as high-quality PNG files (300 DPI)
- **Scalability**: Modular design allows for distributed processing extensions

## Error Handling

The tool includes comprehensive error handling for:
- Malformed timestamp data
- Missing columns
- Database connection issues
- File I/O problems
- Insufficient data for analysis

All errors are logged with detailed information for debugging.

## Logging

The tool uses Python's built-in logging module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # For detailed logs
logging.basicConfig(level=logging.INFO)   # For standard output
logging.basicConfig(level=logging.WARNING) # For warnings only
```

## Contributing

To extend the tool with custom analysis methods:

1. **Add new analyzers**: Inherit from the base analyzer classes
2. **Custom visualizations**: Extend the `Visualizer` class
3. **New data sources**: Add methods to `DataLoader` class
4. **Export formats**: Extend `ResultExporter` class

Example custom analyzer:
```python
class CustomAnalyzer(EventAnalyzer):
    def custom_pattern_detection(self, df):
        # Your custom analysis logic
        return results
```

## Troubleshooting

### Common Issues

**"No module named 'statsmodels'"**
- Install optional dependency: `pip install statsmodels`

**"Timestamp parsing errors"**
- Ensure timestamp column contains valid datetime strings
- Check timezone format consistency

**"Insufficient data for analysis"**
- Verify dataset has enough records (minimum 10-20 time periods)
- Check that timestamp column is correctly specified

**"Memory errors with large datasets"**
- Process data in chunks or filter before analysis
- Consider using more specific time ranges

### Performance Tips

- **Filter early**: Apply filters during preprocessing to reduce data size
- **Choose appropriate intervals**: Use larger intervals (hour/day) for massive datasets
- **Memory monitoring**: Monitor memory usage for very large datasets
- **Batch processing**: Process multiple files separately for better resource management

## License

This project is open source and available under the MIT License.

## Support

For issues, feature requests, or questions:
1. Check the troubleshooting section above
2. Review the example code and configuration options
3. Create detailed issue reports with sample data and error messages

---

**Version**: 1.0.0  
**Python Version**: 3.8+  
**Last Updated**: 2024
