"""
Analysis tools for Ticker Skimmer
Provides visualization and analysis of Reddit data and stock predictions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import requests
import json
from prediction_model import AdaptivePredictionModel
try:
    from config import API_KEY
except ImportError:
    API_KEY = None
    print("Warning: API_KEY not found in config.py. Some functionality will be limited.")

class TickerAnalyzer:
    """
    Analysis and visualization tools for ticker data
    """
    def __init__(self, reddit_data_path='reddit_data_enhanced.csv', api_key=API_KEY):
        self.reddit_data_path = reddit_data_path
        self.api_key = api_key
        self.reddit_data = None
        self.prediction_model = AdaptivePredictionModel(api_key=api_key)
        
    def load_data(self):
        """
        Load Reddit data from CSV
        """
        try:
            df = pd.read_csv(self.reddit_data_path)
            
            # Handle list columns
            for col in ['Title', 'Sentiment_Analysis']:
                if col in df.columns:
                    # Convert string representation of lists to actual lists
                    df[col] = df[col].apply(self._parse_list_column)
                    
            self.reddit_data = df
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing
            self._create_dummy_data()
            return False
            
    def _create_dummy_data(self):
        """
        Create dummy data for testing
        """
        print("Creating dummy data for testing...")
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        data = []
        
        for ticker in tickers:
            data.append({
                'Ticker': ticker,
                'Title': [f"Discussion about {ticker}", f"Why {ticker} is interesting"],
                'Post_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Mentions': np.random.randint(1, 20),
                'Upvotes': np.random.randint(10, 200),
                'Likes_per_mention': np.random.uniform(1, 10),
                'Sentiment_Analysis': [np.random.uniform(-1, 1) for _ in range(3)]
            })
            
        self.reddit_data = pd.DataFrame(data)
        
    def _parse_list_column(self, value):
        """
        Parse a string representation of a list
        """
        if isinstance(value, str):
            # Handle the case where the string is wrapped in brackets
            if value.startswith('[') and value.endswith(']'):
                try:
                    # Try using eval for proper list parsing
                    return eval(value)
                except:
                    # Fallback to simple comma splitting
                    value = value[1:-1]  # Remove brackets
                    return [item.strip() for item in value.split(',')]
            else:
                # Single value, return as a list
                return [value]
        elif isinstance(value, list):
            # Already a list
            return value
        else:
            # Unknown type, return empty list
            return []
    
    def get_top_mentioned_tickers(self, n=10):
        """
        Get the top N mentioned tickers
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        return self.reddit_data.sort_values('Mentions', ascending=False).head(n)
    
    def get_sentiment_rankings(self):
        """
        Get tickers ranked by sentiment
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        # Calculate average sentiment for each ticker
        sentiment_data = []
        for _, row in self.reddit_data.iterrows():
            ticker = row['Ticker']
            sentiment_list = row['Sentiment_Analysis']
            if sentiment_list and len(sentiment_list) > 0:
                avg_sentiment = sum(sentiment_list) / len(sentiment_list)
                sentiment_data.append({
                    'Ticker': ticker,
                    'Average_Sentiment': avg_sentiment,
                    'Mentions': row['Mentions']
                })
                
        # Convert to DataFrame and sort
        sentiment_df = pd.DataFrame(sentiment_data)
        return sentiment_df.sort_values('Average_Sentiment', ascending=False)
    
    def visualize_mentions(self, n=10, save_path=None):
        """
        Visualize the top N mentioned tickers
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        top_n = self.get_top_mentioned_tickers(n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Ticker', y='Mentions', data=top_n)
        plt.title(f'Top {n} Mentioned Tickers')
        plt.xlabel('Ticker')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        return plt
    
    def visualize_sentiment(self, n=10, save_path=None):
        """
        Visualize sentiment for top N tickers
        """
        sentiment_df = self.get_sentiment_rankings()
        if sentiment_df is None or sentiment_df.empty:
            print("Error: No sentiment data available")
            return None
            
        # Get top N by mentions but with sentiment data
        top_n = sentiment_df.sort_values('Mentions', ascending=False).head(n)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(top_n['Ticker'], top_n['Average_Sentiment'])
        
        # Color bars based on sentiment (red for negative, green for positive)
        for i, bar in enumerate(bars):
            if top_n.iloc[i]['Average_Sentiment'] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
                
        plt.title(f'Sentiment for Top {n} Mentioned Tickers')
        plt.xlabel('Ticker')
        plt.ylabel('Average Sentiment Score')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        return plt
    
    def get_stock_data(self, ticker, days=30):
        """
        Get stock data for a ticker
        """
        return self.prediction_model.get_stock_data(ticker, days)
    
    def predict_price_movement(self, ticker):
        """
        Predict price movement for a ticker
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        # Get data for this ticker
        ticker_data = self.reddit_data[self.reddit_data['Ticker'] == ticker]
        if ticker_data.empty:
            return {
                'error': f"No data found for ticker {ticker}"
            }
            
        row = ticker_data.iloc[0]
        sentiment_list = row['Sentiment_Analysis']
        mentions = row['Mentions']
        upvotes = row['Upvotes']
        
        # Prepare features
        features = {
            'sentiment': sum(sentiment_list) / len(sentiment_list) if sentiment_list else 0,
            'mentions': mentions,
            'upvotes': upvotes
        }
        
        # Make prediction
        prediction = self.prediction_model.predict_price_movement(ticker, features)
        
        # Get current stock price
        stock_data = self.get_stock_data(ticker, days=1)
        current_price = None
        if stock_data is not None and not stock_data.empty:
            current_price = stock_data.iloc[0]['close']
            
        # Add current price to prediction
        if current_price:
            prediction['current_price'] = current_price
            prediction['predicted_price'] = current_price * (1 + prediction['price_change_prediction'])
            
        return prediction
    
    def generate_report(self, output_dir='reports'):
        """
        Generate comprehensive analysis report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return False
                
        # Generate timestamp for report
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate visualizations
        mentions_path = f"{output_dir}/top_mentions_{report_time}.png"
        sentiment_path = f"{output_dir}/sentiment_{report_time}.png"
        
        self.visualize_mentions(save_path=mentions_path)
        self.visualize_sentiment(save_path=sentiment_path)
        
        # Get top tickers by mentions
        top_tickers = self.get_top_mentioned_tickers(n=10)
        
        # Generate predictions for top tickers
        predictions = []
        for ticker in top_tickers['Ticker']:
            prediction = self.predict_price_movement(ticker)
            if prediction:
                predictions.append({
                    'ticker': ticker,
                    'prediction': prediction
                })
                
        # Save predictions
        with open(f"{output_dir}/predictions_{report_time}.json", 'w') as f:
            # Convert any non-serializable objects to strings
            json_data = []
            for p in predictions:
                json_p = {'ticker': p['ticker']}
                
                # Handle non-serializable values in prediction
                json_pred = {}
                for k, v in p['prediction'].items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        json_pred[k] = v
                    else:
                        json_pred[k] = str(v)
                
                json_p['prediction'] = json_pred
                json_data.append(json_p)
                
            json.dump(json_data, f, indent=4)
            
        # Create summary report
        top_sentiment = self.get_sentiment_rankings()
        
        summary = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers_analyzed': len(self.reddit_data),
            'total_mentions': self.reddit_data['Mentions'].sum(),
            'most_mentioned': top_tickers.iloc[0]['Ticker'] if not top_tickers.empty else 'N/A',
            'most_mentioned_count': top_tickers.iloc[0]['Mentions'] if not top_tickers.empty else 0,
            'most_positive': top_sentiment.iloc[0]['Ticker'] if not top_sentiment.empty else 'N/A',
            'most_negative': top_sentiment.iloc[-1]['Ticker'] if not top_sentiment.empty and len(top_sentiment) > 0 else 'N/A',
        }
        
        # Save summary
        with open(f"{output_dir}/analysis_summary_{report_time}.txt", 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
                
        # Create HTML report
        html_report = f"""
        <html>
        <head>
            <title>Ticker Skimmer Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Ticker Skimmer Analysis Report</h1>
            <p>Generated on: {summary['report_time']}</p>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Tickers Analyzed</td><td>{summary['tickers_analyzed']}</td></tr>
                <tr><td>Total Mentions</td><td>{summary['total_mentions']}</td></tr>
                <tr><td>Most Mentioned Ticker</td><td>{summary['most_mentioned']} ({summary['most_mentioned_count']} mentions)</td></tr>
                <tr><td>Most Positive Sentiment</td><td>{summary['most_positive']}</td></tr>
                <tr><td>Most Negative Sentiment</td><td>{summary['most_negative']}</td></tr>
            </table>
            
            <h2>Top Mentioned Tickers</h2>
            <img src="top_mentions_{report_time}.png" alt="Top Mentioned Tickers" style="max-width: 100%;">
            
            <h2>Sentiment Analysis</h2>
            <img src="sentiment_{report_time}.png" alt="Sentiment Analysis" style="max-width: 100%;">
            
            <h2>Price Predictions</h2>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Current Price</th>
                    <th>Predicted Change</th>
                    <th>Predicted Price</th>
                    <th>Confidence</th>
                    <th>Market Condition</th>
                </tr>
        """
        
        # Add prediction rows to HTML
        for pred in predictions:
            ticker = pred['ticker']
            p = pred['prediction']
            
            # Check if we have price data
            if 'current_price' in p and 'predicted_price' in p:
                current_price = f"${p['current_price']:.2f}"
                predicted_price = f"${p['predicted_price']:.2f}"
                
                # Format change percentage and determine color
                pct_change = p['price_change_prediction'] * 100
                change_class = 'positive' if pct_change >= 0 else 'negative'
                predicted_change = f"<span class='{change_class}'>{pct_change:.2f}%</span>"
            else:
                current_price = "N/A"
                predicted_price = "N/A"
                predicted_change = "N/A"
                
            # Safely format confidence value
            confidence_value = p.get('confidence', 'N/A')
            if isinstance(confidence_value, (int, float)):
                confidence_display = f"{confidence_value:.2f}"
            else:
                confidence_display = "N/A"
                
            # Add row to table
            html_report += f"""
                <tr>
                    <td>{ticker}</td>
                    <td>{current_price}</td>
                    <td>{predicted_change}</td>
                    <td>{predicted_price}</td>
                    <td>{confidence_display}</td>
                    <td>{p.get('market_condition', 'N/A')}</td>
                </tr>
            """
            
        # Close HTML report
        html_report += """
            </table>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f"{output_dir}/report_{report_time}.html", 'w') as f:
            f.write(html_report)
            
        return f"{output_dir}/report_{report_time}.html"