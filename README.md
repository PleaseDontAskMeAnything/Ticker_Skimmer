# Ticker Skimmer

A powerful Reddit stock analysis tool with Just-In-Time (JIT) optimizations for processing ticker mentions and generating stock predictions.

## Overview

Ticker Skimmer scrapes Reddit (specifically r/wallstreetbets) for stock ticker mentions, analyzes the sentiment and popularity of these mentions, and uses JIT optimization techniques to efficiently process and analyze the data. The tool can predict potential stock price movements based on Reddit sentiment and generate comprehensive reports and visualizations.

## Features

- **Reddit Data Scraping**: Collects ticker mentions, sentiment, and upvotes from Reddit posts
- **JIT Optimization**: Dynamically optimizes frequently executed code paths for better performance
- **Pattern Recognition**: Identifies and optimizes analysis for frequently mentioned tickers
- **Adaptive Prediction Model**: Changes prediction strategies based on market conditions
- **Performance Monitoring**: Tracks execution times and prediction accuracy
- **Interactive Visualizations**: Creates comprehensive dashboards and reports
- **Modular Architecture**: Easily extensible with new features and optimizations

## Project Structure

```
ticker_skimmer/
├── data/                      # Data storage directory
│   └── reddit_data.csv        # Scraped Reddit data
├── reports/                   # Generated reports directory
├── visualizations/            # Generated visualizations directory
├── config.py                  # Configuration and API keys
├── spot.py                    # Ticker watchlist
├── historical.py              # Original data scraping
├── historical_jit.py          # Enhanced data scraping with JIT
├── jit_engine.py              # Core JIT optimization engine
├── pattern_engine.py          # Pattern recognition
├── prediction_model.py        # Adaptive prediction
├── monitor.py                 # Performance monitoring
├── analysis.py                # Data analysis utilities
├── visualization.py           # Interactive visualization utilities
├── main.py                    # Main orchestration script
└── README.md                  # Documentation
```

## Requirements

- Python 3.8+
- praw (Reddit API wrapper)
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- requests
- vaderSentiment

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PleaseDontAskMeAnything/Ticker_Skimmer.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `config.py` file with your API keys:
   ```python
   # Reddit API credentials
   CLIENT_ID = 'your_reddit_client_id'
   CLIENT_SECRET = 'your_reddit_client_secret'
   
   # Alpha Vantage API key
   API_KEY = 'your_alphavantage_api_key'
   ```

## Usage

### Basic Usage

Run the main script with default settings:

```
python main.py
```

This will:
1. Scrape Reddit for ticker mentions
2. Analyze the data
3. Generate predictions
4. Create visualizations and reports

### Command Line Options

```
python main.py --mode [scrape|analyze|full] --limit 2000 --batch 100 --market [neutral|bullish|bearish|volatile] --report-dir reports --input reddit_data.csv --open-report
```

- `--mode`: Operation mode (scrape, analyze, or full)
- `--limit`: Number of Reddit posts to scrape
- `--batch`: Batch size for processing
- `--market`: Market condition for predictions
- `--report-dir`: Directory for reports
- `--input`: Input data file for analysis
- `--open-report`: Open the generated report in browser

### Example Commands

Scrape 1000 Reddit posts:
```
python main.py --mode scrape --limit 1000
```

Analyze existing data with bullish market condition:
```
python main.py --mode analyze --market bullish --input reddit_data.csv
```

## How JIT Optimization Works

Ticker Skimmer uses several JIT optimization techniques:

1. **Execution Tracking**: Monitors how often each code path is executed
2. **Hot Path Detection**: Identifies frequently executed code paths
3. **Compilation Cache**: Stores optimized versions of hot functions
4. **Adaptive Optimization**: Changes optimization strategies based on execution patterns
5. **Pattern Recognition**: Optimizes analysis for frequently mentioned tickers

The JIT engine provides:
- Memoization for frequently called functions
- Batch processing for efficient memory usage
- Optimized data structures for frequent operations
- Performance monitoring and feedback loops

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
