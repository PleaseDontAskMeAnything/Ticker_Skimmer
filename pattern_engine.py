"""
Pattern Engine for Ticker Skimmer
Recognizes patterns in ticker mentions and optimizes analysis for frequently mentioned tickers
"""
import pandas as pd
import numpy as np
from jit_engine import JITEngine, jit_optimized

class TickerPatternEngine:
    """
    Pattern recognition engine that identifies trends in ticker mentions
    and optimizes analysis for frequently occurring patterns
    """
    def __init__(self, historical_data=None):
        self.jit_engine = JITEngine()
        self.historical_data = historical_data if historical_data is not None else {}
        self.pattern_weights = self._initialize_weights()
        self.hot_patterns = set()  # Tracks frequently occurring patterns
        self.pattern_frequency = {}  # Tracks how often each pattern occurs
        self.hot_threshold = 5  # Number of occurrences before a pattern is considered "hot"
        
    def _initialize_weights(self):
        """
        Initialize weights based on historical performance
        """
        weights = {}
        for ticker in self.historical_data:
            # Calculate initial weights based on historical performance or use defaults
            weights[ticker] = {
                'sentiment_weight': 0.5,
                'upvote_weight': 0.3,
                'mention_weight': 0.2
            }
        return weights
    
    def load_historical_data(self, data_path):
        """
        Load historical data from CSV
        """
        try:
            df = pd.read_csv(data_path)
            # Convert DataFrame to dictionary format
            for _, row in df.iterrows():
                ticker = row['Ticker']
                
                # Convert string representation of lists back to actual lists
                sentiment_str = row['Sentiment_Analysis'].replace('[', '').replace(']', '')
                sentiment_list = [float(x.strip()) for x in sentiment_str.split(',') if x.strip()]
                
                title_str = row['Title'].replace('[', '').replace(']', '')
                # Handle potential quotation marks in the titles
                title_list = eval(row['Title']) if '[' in row['Title'] else [row['Title']]
                
                self.historical_data[ticker] = {
                    'Title': title_list,
                    'Post_Date': row['Post_Date'],
                    'Mentions': row['Mentions'],
                    'Upvotes': row['Upvotes'],
                    'Likes_per_mention': row['Likes_per_mention'],
                    'Sentiment_Analysis': sentiment_list
                }
            
            # Re-initialize weights with the new data
            self.pattern_weights = self._initialize_weights()
            return True
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return False
    
    # Fixed the decorator by passing the engine instance
    @jit_optimized(engine=JITEngine(), key_func=lambda self, ticker, *args: f"analyze_{ticker}")
    def analyze_ticker(self, ticker, sentiment, upvotes, mentions):
        """
        Analyze a ticker based on its metrics
        This is JIT-optimized for frequently analyzed tickers
        """
        if ticker in self.hot_patterns:
            # Use optimized path for hot patterns
            return self._optimized_analysis(ticker, sentiment, upvotes, mentions)
        else:
            # Use standard path for cold patterns
            result = self._standard_analysis(ticker, sentiment, upvotes, mentions)
            # Track pattern frequency
            self._track_pattern_frequency(ticker)
            return result
    
    def _optimized_analysis(self, ticker, sentiment, upvotes, mentions):
        """
        Optimized analysis path for frequently occurring tickers
        """
        # Retrieve pre-calculated weights
        weights = self.pattern_weights.get(ticker, {
            'sentiment_weight': 0.5,
            'upvote_weight': 0.3,
            'mention_weight': 0.2
        })
        
        # Fast path calculation
        score = (
            weights['sentiment_weight'] * np.mean(sentiment) +
            weights['upvote_weight'] * np.log1p(upvotes) +
            weights['mention_weight'] * np.log1p(mentions)
        )
        
        # Simple prediction
        if score > 1.5:
            prediction = "strong_buy"
        elif score > 0.5:
            prediction = "buy"
        elif score > -0.5:
            prediction = "hold"
        elif score > -1.5:
            prediction = "sell"
        else:
            prediction = "strong_sell"
            
        return {
            'score': score,
            'prediction': prediction,
            'optimized': True
        }
    
    def _standard_analysis(self, ticker, sentiment, upvotes, mentions):
        """
        Standard analysis path for infrequently occurring tickers
        """
        # Default weights
        default_weights = {
            'sentiment_weight': 0.5,
            'upvote_weight': 0.3,
            'mention_weight': 0.2
        }
        
        # Use default weights if no historical data
        weights = self.pattern_weights.get(ticker, default_weights)
        
        # More detailed calculation for cold path
        avg_sentiment = np.mean(sentiment) if sentiment else 0
        sentiment_factor = avg_sentiment * weights['sentiment_weight']
        
        upvote_factor = np.log1p(upvotes) * weights['upvote_weight'] if upvotes > 0 else 0
        
        mention_factor = np.log1p(mentions) * weights['mention_weight'] if mentions > 0 else 0
        
        # Calculate final score
        score = sentiment_factor + upvote_factor + mention_factor
        
        # More nuanced prediction
        if score > 1.5:
            prediction = "strong_buy"
        elif score > 0.5:
            prediction = "buy"
        elif score > -0.5:
            prediction = "hold"
        elif score > -1.5:
            prediction = "sell"
        else:
            prediction = "strong_sell"
            
        return {
            'score': score,
            'prediction': prediction,
            'sentiment_contribution': sentiment_factor,
            'upvote_contribution': upvote_factor,
            'mention_contribution': mention_factor,
            'optimized': False
        }
    
    def _track_pattern_frequency(self, ticker):
        """
        Track how often a pattern occurs and update hot patterns set
        """
        if ticker not in self.pattern_frequency:
            self.pattern_frequency[ticker] = 0
            
        self.pattern_frequency[ticker] += 1
        
        # Update hot patterns set
        if (self.pattern_frequency[ticker] >= self.hot_threshold and 
            ticker not in self.hot_patterns):
            self.hot_patterns.add(ticker)
            
    def update_weights(self, ticker, prediction_accuracy):
        """
        Update weights based on prediction accuracy
        """
        if ticker in self.pattern_weights:
            current_weights = self.pattern_weights[ticker]
            
            # Simple weight adjustment based on accuracy
            if prediction_accuracy > 0.7:  # Good prediction
                # Slightly increase weights 
                for key in current_weights:
                    current_weights[key] *= 1.05  # 5% increase
            elif prediction_accuracy < 0.3:  # Poor prediction
                # Slightly decrease weights
                for key in current_weights:
                    current_weights[key] *= 0.95  # 5% decrease
                    
            # Normalize weights to sum to 1
            weight_sum = sum(current_weights.values())
            if weight_sum > 0:
                for key in current_weights:
                    current_weights[key] /= weight_sum
                    
            self.pattern_weights[ticker] = current_weights
    
    def get_hot_patterns(self):
        """
        Get the set of frequently occurring patterns
        """
        return self.hot_patterns
    
    def get_pattern_frequency(self):
        """
        Get the frequency of each pattern
        """
        return self.pattern_frequency
    
    def get_stats(self):
        """
        Get statistics about the pattern engine
        """
        return {
            'hot_patterns_count': len(self.hot_patterns),
            'tracked_patterns_count': len(self.pattern_frequency),
            'jit_stats': self.jit_engine.get_stats()
        }