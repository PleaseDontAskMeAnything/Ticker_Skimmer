"""
Main entry point for Ticker Skimmer
Orchestrates the entire data pipeline with JIT optimizations
"""
import os
import argparse
import time
from datetime import datetime
import webbrowser
from historical_jit import get_posts_optimized
from analysis import TickerAnalyzer
from prediction_model import AdaptivePredictionModel
from config import API_KEY

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Ticker Skimmer - Reddit Stock Analysis with JIT')
    
    parser.add_argument('--mode', type=str, default='full', choices=['scrape', 'analyze', 'full'],
                        help='Operation mode: scrape, analyze, or full (default)')
    
    parser.add_argument('--limit', type=int, default=2000,
                        help='Number of Reddit posts to scrape (default: 2000)')
    
    parser.add_argument('--batch', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    
    parser.add_argument('--market', type=str, default='neutral', 
                        choices=['neutral', 'bullish', 'bearish', 'volatile'],
                        help='Market condition for predictions (default: neutral)')
    
    parser.add_argument('--report-dir', type=str, default='reports',
                        help='Directory for reports (default: reports)')
    
    parser.add_argument('--input', type=str, default='reddit_data_enhanced.csv',
                        help='Input data file for analysis (default: reddit_data_enhanced.csv)')
    
    parser.add_argument('--open-report', action='store_true',
                        help='Open the generated report in the default browser')
    
    return parser.parse_args()

def main():
    """
    Main entry point
    """
    args = parse_arguments()
    
    # Create reports directory if it doesn't exist
    os.makedirs(args.report_dir, exist_ok=True)
    
    start_time = datetime.now()
    print(f"Ticker Skimmer starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Operation mode: {args.mode}")
    print(f"Market condition: {args.market}")
    
    # Initialize prediction model with market condition
    prediction_model = AdaptivePredictionModel(api_key=API_KEY)
    prediction_model.set_market_condition(args.market)
    
    if args.mode in ['scrape', 'full']:
        print(f"\n{'=' * 50}")
        print(f"SCRAPING REDDIT DATA")
        print(f"{'=' * 50}")
        print(f"Posts limit: {args.limit}")
        print(f"Batch size: {args.batch}")
        
        # Run the scraper
        stats = get_posts_optimized(limit=args.limit, batch_size=args.batch)
        
        print(f"\nScraping completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        
    if args.mode in ['analyze', 'full']:
        print(f"\n{'=' * 50}")
        print(f"ANALYZING DATA")
        print(f"{'=' * 50}")
        print(f"Input file: {args.input}")
        
        # Initialize analyzer
        analyzer = TickerAnalyzer(reddit_data_path=args.input, api_key=API_KEY)
        
        # Load data
        print("Loading data...")
        if not analyzer.load_data():
            print("Error loading data. Exiting.")
            return
            
        # Generate comprehensive report
        print("Generating report...")
        report_path = analyzer.generate_report(output_dir=args.report_dir)
        
        print(f"Analysis completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        print(f"Report generated: {report_path}")
        
        # Open report in browser if requested
        if args.open_report and report_path:
            print("Opening report in browser...")
            webbrowser.open('file://' + os.path.abspath(report_path))
    
    # Print final summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 50}")
    print(f"PROCESS COMPLETED")
    print(f"{'=' * 50}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Reports saved to: {os.path.abspath(args.report_dir)}")
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nError: {e}")