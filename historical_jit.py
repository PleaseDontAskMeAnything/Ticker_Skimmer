"""
Enhanced version of historical.py with JIT optimizations
"""
import requests
import pandas as pd
import pprint
import praw
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from jit_engine import JITEngine, jit_optimized
from pattern_engine import TickerPatternEngine
from prediction_model import AdaptivePredictionModel
from monitor import PerformanceMonitor
from spot import watchlist, ticker_count
from config import CLIENT_ID, CLIENT_SECRET, API_KEY

# Initialize components
jit_engine = JITEngine()
pattern_engine = TickerPatternEngine()
prediction_model = AdaptivePredictionModel(api_key=API_KEY)
monitor = PerformanceMonitor()

# Initialize Reddit API client
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent="python:ticker_skimmer:v1.0",
        read_only=True
    )
except Exception as e:
    print(f"Error initializing Reddit client: {e}")
    # Create a mock Reddit client for testing purposes
    class MockReddit:
        def __init__(self):
            pass
        def subreddit(self, name):
            return MockSubreddit()
    
    class MockSubreddit:
        def __init__(self):
            pass
        def hot(self, limit=10):
            # Return a list of mock posts
            return [MockPost() for _ in range(min(10, limit))]
    
    class MockPost:
        def __init__(self):
            import random
            self.title = f"Test post about ${random.choice(list(watchlist))} stock"
            self.created_utc = datetime.now().timestamp()
            self.score = random.randint(1, 100)
    
    reddit = MockReddit()

pp = pprint.PrettyPrinter(indent=1)

# Use the jit_engine instance we already created
@jit_optimized(engine=jit_engine, key_func=lambda symbol: f"historic_data_{symbol}")
def get_historic_data(symbol):
    """
    Get historical stock data for a symbol
    """
    start_time = datetime.now()
    
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    r = requests.get(url)
    data = r.json()
    
    execution_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms
    monitor.track_execution(symbol, execution_time)
    
    return data

def process_batch(posts_batch, analyzer):
    """
    Process a batch of Reddit posts
    """
    for post in posts_batch:
        tickers_found = []
        post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        title = post.title
        title_split = title.split()
        sentiment_score = analyzer.polarity_scores(title)["compound"]
        upvotes = post.score
        
        # Use optimized ticker detection for hot tickers
        for token in title_split:
            if token.startswith('$'):
                token = token[1:]
            
            # Check if token is in our watchlist
            if token in watchlist:
                tickers_found.append(token)
                
                # Start execution timing
                start_time = datetime.now()
                
                if token not in ticker_count:
                    ticker_count[token] = {
                        "Ticker": token, 
                        "Title": [title], 
                        "Post_Date": post_date, 
                        "Mentions": 1, 
                        "Upvotes": upvotes, 
                        "Likes_per_mention": 0, 
                        "Sentiment_Analysis": [sentiment_score]
                    }
                else:
                    ticker_count[token]["Mentions"] += 1
                    ticker_count[token]["Title"].append(title)
                    ticker_count[token]["Sentiment_Analysis"].append(sentiment_score)
                
                # End execution timing
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                monitor.track_execution(token, execution_time)
                
                # Check if this ticker is frequently mentioned
                if token in pattern_engine.get_hot_patterns():
                    # Apply pattern analysis for frequently mentioned tickers
                    sentiment_list = ticker_count[token]["Sentiment_Analysis"]
                    upvotes = ticker_count[token]["Upvotes"]
                    mentions = ticker_count[token]["Mentions"]
                    
                    analysis = pattern_engine.analyze_ticker(token, sentiment_list, upvotes, mentions)
                    
                    # Store analysis result in ticker_count
                    if "Analysis" not in ticker_count[token]:
                        ticker_count[token]["Analysis"] = []
                    
                    ticker_count[token]["Analysis"].append(analysis)
                    
                    # Generate prediction
                    features = {
                        'sentiment': sum(sentiment_list) / len(sentiment_list) if sentiment_list else 0,
                        'mentions': mentions,
                        'upvotes': upvotes
                    }
                    
                    prediction = prediction_model.predict_price_movement(token, features)
                    
                    # Store prediction
                    if "Prediction" not in ticker_count[token]:
                        ticker_count[token]["Prediction"] = []
                        
                    ticker_count[token]["Prediction"].append(prediction)

def get_posts_optimized(limit=2000, batch_size=100):
    """
    Get posts from Reddit with JIT optimizations
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # Try to load historical data, but continue if file doesn't exist
    try:
        pattern_engine.load_historical_data('reddit_data.csv')
    except Exception as e:
        print(f"Note: Could not load historical data: {e}")
        print("Continuing with empty history...")
    
    subreddit = reddit.subreddit('wallstreetbets')
    
    # Process in batches for efficient memory usage
    posts_batch = []
    processed_count = 0
    
    print(f"Fetching up to {limit} posts from r/wallstreetbets...")
    
    try:
        for post in subreddit.hot(limit=limit):
            posts_batch.append(post)
            
            if len(posts_batch) >= batch_size:
                process_batch(posts_batch, analyzer)
                processed_count += len(posts_batch)
                print(f"Processed {processed_count} posts...")
                posts_batch = []
                
        # Process any remaining posts
        if posts_batch:
            process_batch(posts_batch, analyzer)
            processed_count += len(posts_batch)
            print(f"Processed {processed_count} posts total.")
            
        # Calculate likes per mention
        for ticker in ticker_count:
            mentions = ticker_count[ticker]["Mentions"]
            if ticker_count[ticker]["Upvotes"] > 0:
                ticker_count[ticker]["Likes_per_mention"] = ticker_count[ticker]["Upvotes"] / mentions
                
        # Save to CSV
        df = pd.DataFrame.from_dict(ticker_count, orient="index")
        df.to_csv('reddit_data_enhanced.csv', index=False)
        
        # Generate performance report
        monitor.generate_report()
        
        print("Data saved to reddit_data_enhanced.csv")
        print("Performance report generated in 'reports' directory")
        
        # Return statistics about our JIT optimizations
        return {
            'jit_engine_stats': jit_engine.get_stats(),
            'pattern_engine_stats': pattern_engine.get_stats(),
            'prediction_model_stats': prediction_model.get_stats(),
            'monitor_stats': monitor.get_optimization_candidates()
        }
        
    except Exception as e:
        print(f"Error processing posts: {e}")
        # Save what we have so far
        df = pd.DataFrame.from_dict(ticker_count, orient="index")
        df.to_csv('reddit_data_partial.csv', index=False)
        print("Partial data saved to reddit_data_partial.csv")
        
        return None

def main():
    """
    Main entry point
    """
    try:
        print("Starting Ticker Skimmer with JIT optimizations...")
        stats = get_posts_optimized()
        if stats:
            print("\nJIT Engine Statistics:")
            pp.pprint(stats['jit_engine_stats'])
            
            print("\nPattern Engine Statistics:")
            pp.pprint(stats['pattern_engine_stats'])
            
            print("\nPrediction Model Statistics:")
            pp.pprint(stats['prediction_model_stats'])
            
            print("\nOptimization Candidates:")
            pp.pprint(stats['monitor_stats'])
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        # Generate report on exit
        monitor.generate_report()
        print("Performance report generated in 'reports' directory")

if __name__ == '__main__':
    main()