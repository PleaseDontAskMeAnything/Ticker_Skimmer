"""
Prediction Model for Ticker Skimmer
Adaptive prediction model with JIT optimization based on market conditions
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from functools import lru_cache
from jit_engine import JITEngine, jit_optimized
import gradio as gr

class AdaptivePredictionModel:
    """
    Adaptive prediction model that uses JIT-like optimization for different market conditions
    and different tickers
    """
    def __init__(self, api_key=None):
        self.jit_engine = JITEngine()
        self.models = {}  # Different models for different market conditions
        self.accuracy_tracker = {}
        self.compilation_cache = {}
        self.market_condition = "neutral"  # Default market condition
        self.api_key = api_key  # Alpha Vantage API key
        
    def set_api_key(self, api_key):
        """
        Set the Alpha Vantage API key
        """
        self.api_key = api_key
        
    def get_stock_data(self, ticker, days=30):
        """
        Get historical stock data for a ticker
        """
        if not self.api_key:
            raise ValueError("API key not set. Use set_api_key() to set it.")
            
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={self.api_key}'
        try:
            response = requests.get(url)
            data = response.json()
            
            # Extract time series data
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index(ascending=False)
                
                # Convert columns to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                    
                # Rename columns
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Limit to requested number of days
                return df.head(days)
            else:
                print(f"No time series data found for {ticker}")
                return None
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            # Return mock data for testing
            dates = [datetime.now() - timedelta(days=i) for i in range(days)]
            mock_data = {
                'open': np.random.uniform(90, 110, days),
                'high': np.random.uniform(95, 115, days),
                'low': np.random.uniform(85, 105, days),
                'close': np.random.uniform(90, 110, days),
                'volume': np.random.randint(1000, 10000, days)
            }
            df = pd.DataFrame(mock_data, index=dates)
            return df
    
    def _detect_market_condition(self):
        """
        Detect the current market condition based on recent market data
        """
        # This would typically involve analyzing market-wide indicators
        # For simplicity, we'll return a fixed value for now
        return self.market_condition
    
    def set_market_condition(self, condition):
        """
        Manually set the market condition
        """
        valid_conditions = ["volatile", "bullish", "bearish", "neutral"]
        if condition not in valid_conditions:
            raise ValueError(f"Invalid market condition. Must be one of {valid_conditions}")
            
        self.market_condition = condition
        
    # Fixed the decorator by passing the engine instance
    @jit_optimized(engine=JITEngine(), key_func=lambda self, ticker, *args: f"predict_{ticker}_{self._detect_market_condition()}")
    def predict_price_movement(self, ticker, features):
        """
        Predict price movement for a ticker based on features and market condition
        This function is JIT-optimized for frequently predicted tickers
        """
        market_condition = self._detect_market_condition()
        
        # JIT compilation - get or generate optimized prediction code
        key = (ticker, market_condition)
        if key not in self.compilation_cache:
            # Compile a model specific to this ticker and market condition
            self.compilation_cache[key] = self._compile_prediction_model(ticker, market_condition)
        
        # Execute the compiled prediction function
        prediction_func = self.compilation_cache[key]
        return prediction_func(features)
    
    def _compile_prediction_model(self, ticker, market_condition):
        """
        Create an optimized function based on historical performance
        This simulates JIT compilation by selecting the most effective algorithm
        """
        if market_condition == "volatile":
            return lambda features: self._volatile_market_prediction(features)
        elif market_condition == "bullish":
            return lambda features: self._bullish_market_prediction(features)
        elif market_condition == "bearish":
            return lambda features: self._bearish_market_prediction(features)
        else:
            return lambda features: self._default_prediction(features)
    
    def _volatile_market_prediction(self, features):
        """
        Prediction algorithm for volatile markets
        """
        sentiment = features.get('sentiment', 0)
        mentions = features.get('mentions', 0)
        upvotes = features.get('upvotes', 0)
        
        # In volatile markets, sentiment has higher impact
        sentiment_factor = sentiment * 0.6
        mention_factor = np.log1p(mentions) * 0.3
        upvote_factor = np.log1p(upvotes) * 0.1
        
        score = sentiment_factor + mention_factor + upvote_factor
        
        return {
            'price_change_prediction': score * 0.05,  # Estimated % change
            'confidence': min(0.9, abs(score) / 5),  # Confidence level
            'market_condition': 'volatile'
        }
    
    def _bullish_market_prediction(self, features):
        """
        Prediction algorithm for bullish markets
        """
        sentiment = features.get('sentiment', 0)
        mentions = features.get('mentions', 0)
        upvotes = features.get('upvotes', 0)
        
        # In bullish markets, positive sentiment has even higher impact
        sentiment_factor = max(0, sentiment) * 0.7 + min(0, sentiment) * 0.3
        mention_factor = np.log1p(mentions) * 0.2
        upvote_factor = np.log1p(upvotes) * 0.1
        
        score = sentiment_factor + mention_factor + upvote_factor
        
        return {
            'price_change_prediction': score * 0.03 + 0.01,  # Bullish bias
            'confidence': min(0.85, abs(score) / 5),
            'market_condition': 'bullish'
        }
    
    def _bearish_market_prediction(self, features):
        """
        Prediction algorithm for bearish markets
        """
        sentiment = features.get('sentiment', 0)
        mentions = features.get('mentions', 0)
        upvotes = features.get('upvotes', 0)
        
        # In bearish markets, negative sentiment has higher impact
        sentiment_factor = max(0, sentiment) * 0.3 + min(0, sentiment) * 0.7
        mention_factor = np.log1p(mentions) * 0.2
        upvote_factor = np.log1p(upvotes) * 0.1
        
        score = sentiment_factor + mention_factor + upvote_factor
        
        return {
            'price_change_prediction': score * 0.03 - 0.01,  # Bearish bias
            'confidence': min(0.85, abs(score) / 5),
            'market_condition': 'bearish'
        }
    
    def _default_prediction(self, features):
        """
        Default prediction algorithm for neutral markets
        """
        sentiment = features.get('sentiment', 0)
        mentions = features.get('mentions', 0)
        upvotes = features.get('upvotes', 0)
        
        # Balanced approach
        sentiment_factor = sentiment * 0.5
        mention_factor = np.log1p(mentions) * 0.3
        upvote_factor = np.log1p(upvotes) * 0.2
        
        score = sentiment_factor + mention_factor + upvote_factor
        
        return {
            'price_change_prediction': score * 0.02,
            'confidence': min(0.8, abs(score) / 5),
            'market_condition': 'neutral'
        }
    
    def feedback_loop(self, ticker, prediction, actual_movement):
        """
        Update model based on feedback from actual price movements
        """
        if ticker not in self.accuracy_tracker:
            self.accuracy_tracker[ticker] = []
            
        # Calculate prediction accuracy
        predicted = prediction['price_change_prediction']
        error = abs(predicted - actual_movement)
        max_error = max(abs(predicted), abs(actual_movement), 0.01)  # Avoid division by zero
        accuracy = 1 - (error / max_error)
        accuracy = max(0, min(1, accuracy))  # Clamp to [0, 1]
        
        # Track accuracy
        self.accuracy_tracker[ticker].append(accuracy)
        
        # If accuracy is consistently low, invalidate cached model
        if len(self.accuracy_tracker[ticker]) >= 5:
            recent_accuracy = np.mean(self.accuracy_tracker[ticker][-5:])
            if recent_accuracy < 0.3:
                # Poor accuracy, remove from cache to force recompilation
                market_condition = self._detect_market_condition()
                key = (ticker, market_condition)
                if key in self.compilation_cache:
                    del self.compilation_cache[key]
                    
                # Reset accuracy tracker for this ticker
                self.accuracy_tracker[ticker] = []
                
        return accuracy
    
    def get_stats(self):
        """
        Get statistics about the prediction model
        """
        return {
            'market_condition': self.market_condition,
            'cached_models': len(self.compilation_cache),
            'accuracy_tracked_tickers': len(self.accuracy_tracker),
            'jit_stats': self.jit_engine.get_stats()
        }