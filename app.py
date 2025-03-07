"""
Gradio Web UI for Ticker Skimmer
Provides a user-friendly interface for the Ticker Skimmer tool
"""
import gradio as gr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Ticker Skimmer components
from historical_jit import get_posts_optimized
from analysis import TickerAnalyzer
from prediction_model import AdaptivePredictionModel
try:
    from config import API_KEY
except ImportError:
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your Alpha Vantage API key

# Initialize components
analyzer = TickerAnalyzer(reddit_data_path='reddit_data_enhanced.csv', api_key=API_KEY)
prediction_model = AdaptivePredictionModel(api_key=API_KEY)

# Create necessary directories
os.makedirs('reports', exist_ok=True)
os.makedirs('data', exist_ok=True)


def scrape_reddit(num_posts, batch_size, progress=gr.Progress()):
    """
    Scrape Reddit for ticker mentions with progress updates
    """
    progress(0, desc="Starting Reddit scraper...")
    
    # Validate inputs
    num_posts = max(10, min(5000, int(num_posts)))
    batch_size = max(10, min(500, int(batch_size)))
    
    # Start scraping
    progress(0.1, desc=f"Scraping {num_posts} posts in batches of {batch_size}...")
    
    try:
        stats = get_posts_optimized(limit=num_posts, batch_size=batch_size)
        
        # Load the resulting data
        progress(0.8, desc="Processing scraped data...")
        df = pd.read_csv('reddit_data_enhanced.csv')
        
        # Generate a summary
        summary = {
            "Total tickers found": len(df),
            "Total mentions": df['Mentions'].sum() if 'Mentions' in df.columns else 0,
            "Most mentioned ticker": df.iloc[0]['Ticker'] if not df.empty else "None",
            "Scraping completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create a simple visualization
        progress(0.9, desc="Creating visualization...")
        fig = create_mentions_chart(df)
        
        progress(1.0, desc="Scraping completed!")
        return json.dumps(summary, indent=2), fig, "Scraping completed successfully!"
        
    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        return json.dumps({"Error": error_msg}, indent=2), None, error_msg


def create_mentions_chart(df, n=10):
    """
    Create a chart of top mentioned tickers
    """
    # Sort and get top N
    if 'Mentions' in df.columns:
        top_n = df.sort_values('Mentions', ascending=False).head(n)
        
        # Create figure
        fig = px.bar(
            top_n,
            x='Ticker',
            y='Mentions',
            title=f'Top {n} Mentioned Tickers',
            labels={'Mentions': 'Number of Mentions', 'Ticker': 'Ticker Symbol'}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Ticker',
            yaxis_title='Number of Mentions',
            template='plotly_white'
        )
        
        return fig
    else:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No mention data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def generate_analysis_report():
    """
    Generate a full analysis report
    """
    try:
        # Generate report
        report_path = analyzer.generate_report(output_dir='reports')
        
        if report_path and os.path.exists(report_path):
            with open(report_path, 'r') as f:
                html_content = f.read()
                
            # Extract images
            report_time = os.path.basename(report_path).replace('report_', '').replace('.html', '')
            mentions_img = f"reports/top_mentions_{report_time}.png"
            sentiment_img = f"reports/sentiment_{report_time}.png"
            
            if os.path.exists(mentions_img) and os.path.exists(sentiment_img):
                return html_content, mentions_img, sentiment_img, "Report generated successfully!"
            else:
                return html_content, None, None, "Report generated but images not found."
        else:
            return "<p>Failed to generate report</p>", None, None, "Failed to generate report."
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        return f"<p>{error_msg}</p>", None, None, error_msg


def predict_ticker(ticker, market_condition):
    """
    Generate prediction for a specific ticker
    """
    try:
        # Set market condition
        prediction_model.set_market_condition(market_condition)
        
        # Get prediction
        prediction = analyzer.predict_price_movement(ticker)
        
        if prediction and 'error' not in prediction:
            # Create a nice display of the prediction
            prediction_html = f"""
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px;">
                <h3>Prediction for {ticker}</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Metric</th>
                        <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Value</th>
                    </tr>
            """
            
            # Current price
            if 'current_price' in prediction:
                prediction_html += f"""
                    <tr>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Current Price</td>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">${prediction['current_price']:.2f}</td>
                    </tr>
                """
            
            # Predicted change
            if 'price_change_prediction' in prediction:
                change = prediction['price_change_prediction'] * 100
                color = "green" if change >= 0 else "red"
                prediction_html += f"""
                    <tr>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Predicted Change</td>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd; color: {color};">{change:.2f}%</td>
                    </tr>
                """
            
            # Predicted price
            if 'predicted_price' in prediction:
                prediction_html += f"""
                    <tr>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Predicted Price</td>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">${prediction['predicted_price']:.2f}</td>
                    </tr>
                """
            
            # Confidence
            if 'confidence' in prediction:
                prediction_html += f"""
                    <tr>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Confidence</td>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{prediction['confidence']:.2f}</td>
                    </tr>
                """
            
            # Market condition
            if 'market_condition' in prediction:
                prediction_html += f"""
                    <tr>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Market Condition</td>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{prediction['market_condition']}</td>
                    </tr>
                """
            
            prediction_html += """
                </table>
            </div>
            """
            
            # Try to get stock data for a chart
            stock_data = analyzer.get_stock_data(ticker, days=30)
            
            if stock_data is not None and not stock_data.empty:
                # Create a stock price chart
                fig = go.Figure()
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['open'],
                        high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        name='Price'
                    )
                )
                
                # Add prediction marker
                if 'current_price' in prediction and 'predicted_price' in prediction:
                    # Get the last date and add a prediction point
                    last_date = stock_data.index[-1]
                    pred_date = last_date + pd.Timedelta(days=1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, pred_date],
                            y=[prediction['current_price'], prediction['predicted_price']],
                            mode='lines+markers',
                            line=dict(color='red', width=2, dash='dash'),
                            name='Prediction'
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Price with Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_white'
                )
                
                # Convert to HTML
                chart_html = fig.to_html(full_html=False)
                
                return prediction_html + chart_html, f"Prediction for {ticker} generated successfully!"
            else:
                return prediction_html, f"Prediction for {ticker} generated successfully (no price chart available)."
        else:
            error_msg = prediction.get('error', 'No data found for this ticker') if prediction else 'Failed to generate prediction'
            return f"<p>Error: {error_msg}</p>", error_msg
    except Exception as e:
        error_msg = f"Error generating prediction: {str(e)}"
        return f"<p>{error_msg}</p>", error_msg


def load_ticker_list():
    """
    Load the list of tickers from spot.py
    """
    try:
        from spot import watchlist
        return list(watchlist)
    except ImportError:
        # Fallback to a default list
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]


# Create the Gradio interface
with gr.Blocks(title="Ticker Skimmer", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 📈 Ticker Skimmer
        
        **Analyze stock ticker mentions from Reddit with JIT optimization**
        
        This tool scrapes Reddit for stock ticker mentions, analyzes sentiment, and generates price predictions.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("📊 Scrape Reddit"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Scrape Reddit for Ticker Mentions")
                    
                    num_posts = gr.Slider(
                        minimum=10, 
                        maximum=5000, 
                        value=200, 
                        step=10, 
                        label="Number of Posts to Scrape",
                        info="More posts will take longer to process"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=10, 
                        maximum=500, 
                        value=50, 
                        step=10, 
                        label="Batch Size",
                        info="How many posts to process at once"
                    )
                    
                    scrape_btn = gr.Button("Start Scraping", variant="primary")
                    
                    scrape_output = gr.JSON(label="Scraping Results")
                    scrape_status = gr.Textbox(label="Status")
                
                with gr.Column(scale=3):
                    chart_output = gr.Plot(label="Top Mentioned Tickers")
            
            scrape_btn.click(
                scrape_reddit, 
                inputs=[num_posts, batch_size], 
                outputs=[scrape_output, chart_output, scrape_status]
            )
            
        with gr.TabItem("📈 Analysis Report"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Generate Full Analysis Report")
                    generate_btn = gr.Button("Generate Report", variant="primary")
                    report_status = gr.Textbox(label="Status")
                    
            with gr.Row():
                with gr.Column(scale=3):
                    report_html = gr.HTML(label="Analysis Report")
                
                with gr.Column(scale=2):
                    mentions_img = gr.Image(label="Top Mentions", show_label=True)
                    sentiment_img = gr.Image(label="Sentiment Analysis", show_label=True)
            
            generate_btn.click(
                generate_analysis_report, 
                inputs=[], 
                outputs=[report_html, mentions_img, sentiment_img, report_status]
            )
            
        with gr.TabItem("🔮 Ticker Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Predict Price Movement for a Ticker")
                    
                    ticker_dropdown = gr.Dropdown(
                        choices=load_ticker_list(), 
                        label="Select Ticker",
                        info="Choose a stock ticker to analyze",
                        allow_custom_value=True
                    )
                    
                    market_condition = gr.Radio(
                        choices=["neutral", "bullish", "bearish", "volatile"],
                        value="neutral",
                        label="Market Condition",
                        info="Different market conditions affect prediction models"
                    )
                    
                    predict_btn = gr.Button("Generate Prediction", variant="primary")
                    prediction_status = gr.Textbox(label="Status")
                    
                with gr.Column(scale=3):
                    prediction_output = gr.HTML(label="Prediction Results")
            
            predict_btn.click(
                predict_ticker, 
                inputs=[ticker_dropdown, market_condition], 
                outputs=[prediction_output, prediction_status]
            )
            
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                ## About Ticker Skimmer
                
                Ticker Skimmer is a powerful stock analysis tool that uses Just-In-Time (JIT) optimization 
                to efficiently process Reddit ticker mentions and generate stock price predictions.
                
                ### Features:
                
                - **Reddit Data Scraping**: Collects and analyzes ticker mentions, sentiment scores, and upvotes from social media
                - **JIT Optimization Engine**: Dynamically optimizes frequent execution paths for better performance
                - **Pattern Recognition**: Identifies frequently mentioned tickers and optimizes their analysis
                - **Adaptive Prediction Model**: Changes prediction strategies based on market conditions
                - **Performance Monitoring**: Tracks execution times and prediction accuracy
                
                ### How It Works:
                
                1. **Data Collection**: Scrapes Reddit posts for ticker mentions and sentiment
                2. **JIT Optimization**: Tracks execution patterns and optimizes frequent paths
                3. **Pattern Analysis**: Applies specialized analysis to frequently mentioned tickers
                4. **Prediction Generation**: Creates price movement predictions based on sentiment, mentions, and market conditions
                
                For more information, visit the [GitHub repository](https://github.com/yourusername/ticker_skimmer).
                """
            )

if __name__ == "__main__":
    app.launch(share=True)
                                                