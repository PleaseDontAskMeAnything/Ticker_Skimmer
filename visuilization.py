"""
Visualization utilities for Ticker Skimmer
Provides interactive visualization of Reddit data and stock predictions
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

class TickerVisualizer:
    """
    Interactive visualization tools for ticker data
    """
    def __init__(self, reddit_data_path='reddit_data_enhanced.csv'):
        self.reddit_data_path = reddit_data_path
        self.reddit_data = None
        self.output_dir = 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
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
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(self._parse_list_column)
                    
            self.reddit_data = df
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
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
    
    def create_mentions_bar_chart(self, n=20, save=True, show=True):
        """
        Create an interactive bar chart of top mentioned tickers
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
        
        # Get top N mentioned tickers
        top_n = self.reddit_data.sort_values('Mentions', ascending=False).head(n)
        
        # Calculate average sentiment
        sentiments = []
        for _, row in top_n.iterrows():
            sentiment_list = row['Sentiment_Analysis']
            avg_sentiment = sum(sentiment_list) / len(sentiment_list) if sentiment_list else 0
            sentiments.append(avg_sentiment)
            
        top_n['Avg_Sentiment'] = sentiments
        
        # Create interactive bar chart with Plotly
        fig = px.bar(
            top_n,
            x='Ticker',
            y='Mentions',
            color='Avg_Sentiment',
            color_continuous_scale='RdYlGn',
            title=f'Top {n} Mentioned Tickers',
            labels={'Mentions': 'Number of Mentions', 'Ticker': 'Ticker Symbol', 'Avg_Sentiment': 'Average Sentiment'},
            hover_data=['Upvotes', 'Likes_per_mention', 'Avg_Sentiment']
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Ticker',
            yaxis_title='Number of Mentions',
            coloraxis_colorbar=dict(title='Sentiment'),
            template='plotly_white'
        )
        
        # Save if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.output_dir, f'mentions_bar_chart_{timestamp}.html')
            fig.write_html(file_path)
            print(f"Saved mentions bar chart to {file_path}")
            
        # Show if requested
        if show:
            fig.show()
            
        return fig
    
    def create_sentiment_heatmap(self, save=True, show=True):
        """
        Create a heatmap of sentiment scores
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        # Prepare data for heatmap
        sentiment_data = []
        for _, row in self.reddit_data.iterrows():
            ticker = row['Ticker']
            sentiment_list = row['Sentiment_Analysis']
            if sentiment_list and len(sentiment_list) > 0:
                for sentiment in sentiment_list:
                    sentiment_data.append({
                        'Ticker': ticker,
                        'Sentiment': sentiment
                    })
                    
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Create pivot table for heatmap
        pivot = sentiment_df.pivot_table(
            values='Sentiment',
            index='Ticker',
            aggfunc=['mean', 'count', 'min', 'max']
        ).sort_values(('count', 'Sentiment'), ascending=False).head(20)
        
        # Flatten the MultiIndex columns
        pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]
        
        # Reset index to make Ticker a column
        pivot_reset = pivot.reset_index()
        
        # Create interactive heatmap
        fig = px.imshow(
            pivot_reset.set_index('Ticker')[['mean_Sentiment']].T,
            color_continuous_scale='RdYlGn',
            labels=dict(color="Sentiment Score"),
            title='Sentiment Heatmap by Ticker'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Ticker',
            yaxis_title='Metric',
            coloraxis_colorbar=dict(title='Sentiment Score'),
            template='plotly_white'
        )
        
        # Save if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.output_dir, f'sentiment_heatmap_{timestamp}.html')
            fig.write_html(file_path)
            print(f"Saved sentiment heatmap to {file_path}")
            
        # Show if requested
        if show:
            fig.show()
            
        return fig
    
    def create_mention_sentiment_scatter(self, save=True, show=True):
        """
        Create a scatter plot of mentions vs sentiment
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        # Calculate average sentiment for each ticker
        sentiment_data = []
        for _, row in self.reddit_data.iterrows():
            ticker = row['Ticker']
            mentions = row['Mentions']
            upvotes = row['Upvotes']
            sentiment_list = row['Sentiment_Analysis']
            if sentiment_list and len(sentiment_list) > 0:
                avg_sentiment = sum(sentiment_list) / len(sentiment_list)
                sentiment_data.append({
                    'Ticker': ticker,
                    'Mentions': mentions,
                    'Upvotes': upvotes,
                    'Avg_Sentiment': avg_sentiment
                })
                
        # Convert to DataFrame
        scatter_df = pd.DataFrame(sentiment_data)
        
        # Create scatter plot
        fig = px.scatter(
            scatter_df,
            x='Mentions',
            y='Avg_Sentiment',
            size='Upvotes',
            color='Avg_Sentiment',
            color_continuous_scale='RdYlGn',
            hover_name='Ticker',
            title='Mentions vs Sentiment by Ticker',
            labels={
                'Mentions': 'Number of Mentions',
                'Avg_Sentiment': 'Average Sentiment',
                'Upvotes': 'Total Upvotes'
            },
            size_max=50
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            xaxis_title='Number of Mentions',
            yaxis_title='Average Sentiment',
            coloraxis_colorbar=dict(title='Sentiment'),
            template='plotly_white'
        )
        
        # Save if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.output_dir, f'mention_sentiment_scatter_{timestamp}.html')
            fig.write_html(file_path)
            print(f"Saved scatter plot to {file_path}")
            
        # Show if requested
        if show:
            fig.show()
            
        return fig
    
    def create_dashboard(self, stock_data=None, save=True):
        """
        Create a comprehensive dashboard with multiple visualizations
        """
        if self.reddit_data is None:
            self.load_data()
            if self.reddit_data is None:
                return None
                
        # Create a subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top Mentioned Tickers',
                'Sentiment by Ticker',
                'Mentions vs Sentiment',
                'Upvotes vs Mentions'
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Top Mentioned Tickers (Bar Chart)
        top_n = self.reddit_data.sort_values('Mentions', ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=top_n['Ticker'],
                y=top_n['Mentions'],
                name='Mentions',
                marker_color='royalblue'
            ),
            row=1, col=1
        )
        
        # 2. Sentiment by Ticker (Heatmap)
        # Calculate average sentiment for each ticker in top_n
        sentiment_values = []
        tickers = []
        for _, row in top_n.iterrows():
            ticker = row['Ticker']
            sentiment_list = row['Sentiment_Analysis']
            if sentiment_list and len(sentiment_list) > 0:
                avg_sentiment = sum(sentiment_list) / len(sentiment_list)
                sentiment_values.append(avg_sentiment)
                tickers.append(ticker)
                
        fig.add_trace(
            go.Heatmap(
                z=[sentiment_values],
                x=tickers,
                y=['Sentiment'],
                colorscale='RdYlGn',
                showscale=True
            ),
            row=1, col=2
        )
        
        # 3. Mentions vs Sentiment (Scatter Plot)
        # Calculate average sentiment for each ticker
        sentiment_data = []
        for _, row in self.reddit_data.iterrows():
            ticker = row['Ticker']
            mentions = row['Mentions']
            sentiment_list = row['Sentiment_Analysis']
            if sentiment_list and len(sentiment_list) > 0:
                avg_sentiment = sum(sentiment_list) / len(sentiment_list)
                sentiment_data.append({
                    'Ticker': ticker,
                    'Mentions': mentions,
                    'Avg_Sentiment': avg_sentiment
                })
                
        scatter_df = pd.DataFrame(sentiment_data)
        
        fig.add_trace(
            go.Scatter(
                x=scatter_df['Mentions'],
                y=scatter_df['Avg_Sentiment'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=scatter_df['Avg_Sentiment'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                text=scatter_df['Ticker'],
                hoverinfo='text+x+y'
            ),
            row=2, col=1
        )
        
        # 4. Upvotes vs Mentions (Scatter Plot)
        fig.add_trace(
            go.Scatter(
                x=self.reddit_data['Mentions'],
                y=self.reddit_data['Upvotes'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='royalblue',
                    opacity=0.7
                ),
                text=self.reddit_data['Ticker'],
                hoverinfo='text+x+y'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Ticker Skimmer Dashboard",
            height=900,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Ticker", row=1, col=1)
        fig.update_yaxes(title_text="Mentions", row=1, col=1)
        
        fig.update_xaxes(title_text="Ticker", row=1, col=2)
        
        fig.update_xaxes(title_text="Mentions", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment", row=2, col=1)
        
        fig.update_xaxes(title_text="Mentions", row=2, col=2)
        fig.update_yaxes(title_text="Upvotes", row=2, col=2)
        
        # Save if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.output_dir, f'dashboard_{timestamp}.html')
            fig.write_html(file_path)
            print(f"Saved dashboard to {file_path}")
            
        return fig
    
    def visualize_jit_performance(self, performance_log_path, save=True, show=True):
        """
        Visualize JIT performance metrics from log file
        """
        try:
            # Load performance log
            perf_df = pd.read_csv(performance_log_path)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Execution Time by Ticker', 'Optimization Status'),
                vertical_spacing=0.2
            )
            
            # Plot execution time by ticker
            for ticker, group in perf_df.groupby('ticker'):
                fig.add_trace(
                    go.Scatter(
                        x=range(len(group)),
                        y=group['execution_time'],
                        mode='lines+markers',
                        name=ticker
                    ),
                    row=1, col=1
                )
                
            # Plot optimization status
            optimized_tickers = perf_df[perf_df['needs_optimization'] == True]['ticker'].unique()
            
            for ticker in optimized_tickers:
                ticker_data = perf_df[perf_df['ticker'] == ticker]
                optimization_points = ticker_data[ticker_data['needs_optimization'] == True]
                
                # Add vertical lines where optimization occurs
                for idx, point in optimization_points.iterrows():
                    fig.add_vline(
                        x=idx, line_dash="dash", line_color="red",
                        row=1, col=1
                    )
            
            # Create bar chart of optimized vs non-optimized tickers
            all_tickers = perf_df['ticker'].unique()
            non_optimized = [t for t in all_tickers if t not in optimized_tickers]
            
            fig.add_trace(
                go.Bar(
                    x=['Optimized', 'Non-Optimized'],
                    y=[len(optimized_tickers), len(non_optimized)],
                    marker_color=['red', 'blue']
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title_text="JIT Performance Analysis",
                height=800,
                width=1000,
                template='plotly_white'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Execution Sequence", row=1, col=1)
            fig.update_yaxes(title_text="Execution Time (ms)", row=1, col=1)
            
            fig.update_xaxes(title_text="Optimization Status", row=2, col=1)
            fig.update_yaxes(title_text="Number of Tickers", row=2, col=1)
            
            # Save if requested
            if save:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = os.path.join(self.output_dir, f'jit_performance_{timestamp}.html')
                fig.write_html(file_path)
                print(f"Saved JIT performance visualization to {file_path}")
                
            # Show if requested
            if show:
                fig.show()
                
            return fig
        except Exception as e:
            print(f"Error visualizing JIT performance: {e}")
            return None