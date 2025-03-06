"""
Performance Monitor for Ticker Skimmer
Tracks execution performance and triggers JIT optimizations as needed
"""
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import numpy as np

class PerformanceMonitor:
    """
    Monitors performance metrics and triggers optimizations when needed
    """
    def __init__(self):
        self.execution_times = {}
        self.prediction_accuracy = {}
        self.optimization_thresholds = {
            'execution_time': 100,  # ms
            'accuracy_drop': 0.05   # 5% drop in accuracy
        }
        self.log_data = []
        self.start_time = datetime.now()
        self.optimization_events = []
        
    def track_execution(self, ticker, execution_time):
        """
        Track execution time for a ticker
        Returns True if optimization is needed
        """
        if ticker not in self.execution_times:
            self.execution_times[ticker] = []
            
        self.execution_times[ticker].append(execution_time)
        
        # Log this execution
        self.log_data.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'execution_time': execution_time,
            'needs_optimization': self._needs_optimization(ticker)
        })
        
        # Check if optimization is needed
        needs_opt = self._needs_optimization(ticker)
        if needs_opt:
            self.optimization_events.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': ticker,
                'reason': 'Execution time exceeded threshold',
                'threshold': self.optimization_thresholds['execution_time'],
                'current_value': execution_time
            })
            
        return needs_opt
    
    def track_prediction(self, ticker, prediction, actual):
        """
        Track prediction accuracy for a ticker
        """
        if ticker not in self.prediction_accuracy:
            self.prediction_accuracy[ticker] = []
            
        # Calculate a simple accuracy metric (0 to 1)
        error = abs(prediction - actual)
        max_val = max(abs(prediction), abs(actual), 0.01)  # Avoid division by zero
        accuracy = 1 - (error / max_val)
        accuracy = max(0, min(1, accuracy))  # Clamp to [0, 1]
        
        self.prediction_accuracy[ticker].append(accuracy)
        
        # Check if accuracy has dropped significantly
        if len(self.prediction_accuracy[ticker]) >= 3:
            recent_accuracy = np.mean(self.prediction_accuracy[ticker][-3:])
            if len(self.prediction_accuracy[ticker]) >= 6:
                previous_accuracy = np.mean(self.prediction_accuracy[ticker][-6:-3])
                accuracy_drop = previous_accuracy - recent_accuracy
                
                if accuracy_drop > self.optimization_thresholds['accuracy_drop']:
                    self.optimization_events.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'ticker': ticker,
                        'reason': 'Accuracy drop exceeded threshold',
                        'threshold': self.optimization_thresholds['accuracy_drop'],
                        'current_value': accuracy_drop
                    })
        
        # Log this prediction
        self.log_data.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'prediction': prediction,
            'actual': actual,
            'accuracy': accuracy
        })
        
        return accuracy
            
    def _needs_optimization(self, ticker):
        """
        Check if a ticker needs optimization based on execution time
        """
        if ticker not in self.execution_times:
            return False
            
        # Need at least a few data points to make a decision
        if len(self.execution_times[ticker]) < 3:
            return False
            
        # Check if average execution time exceeds threshold
        avg_time = sum(self.execution_times[ticker][-3:]) / 3
        if avg_time > self.optimization_thresholds['execution_time']:
            return True
            
        return False
    
    def get_optimization_candidates(self):
        """
        Get a list of tickers that are candidates for optimization
        """
        candidates = []
        for ticker, times in self.execution_times.items():
            if len(times) >= 3:
                avg_time = sum(times[-3:]) / 3
                if avg_time > self.optimization_thresholds['execution_time']:
                    candidates.append({
                        'ticker': ticker,
                        'avg_execution_time': avg_time,
                        'executions': len(times),
                        'recent_times': times[-3:]
                    })
                    
        return sorted(candidates, key=lambda x: x['avg_execution_time'], reverse=True)
    
    def get_accuracy_stats(self):
        """
        Get accuracy statistics for all tracked tickers
        """
        stats = {}
        for ticker, accuracies in self.prediction_accuracy.items():
            if len(accuracies) > 0:
                stats[ticker] = {
                    'mean_accuracy': sum(accuracies) / len(accuracies),
                    'latest_accuracy': accuracies[-1],
                    'num_predictions': len(accuracies),
                    'trend': 'improving' if len(accuracies) >= 2 and accuracies[-1] > accuracies[-2] else 'declining'
                }
                
        return stats
    
    def set_optimization_threshold(self, metric, value):
        """
        Set a new threshold for optimization
        """
        if metric in self.optimization_thresholds:
            self.optimization_thresholds[metric] = value
            return True
        return False
    
    def save_logs(self, filename='performance_logs.csv'):
        """
        Save performance logs to CSV
        """
        if not self.log_data:
            return False
            
        df = pd.DataFrame(self.log_data)
        df.to_csv(filename, index=False)
        
        # Also save optimization events
        if self.optimization_events:
            events_df = pd.DataFrame(self.optimization_events)
            events_df.to_csv(filename.replace('.csv', '_events.csv'), index=False)
            
        return True
    
    def generate_report(self, output_dir='reports'):
        """
        Generate performance report with visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a DataFrame from log data
        if not self.log_data:
            return False
            
        df = pd.DataFrame(self.log_data)
        
        # Generate report timestamp
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        df.to_csv(f"{output_dir}/performance_log_{report_time}.csv", index=False)
        
        # Generate execution time visualization
        if 'execution_time' in df.columns and 'ticker' in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Group by ticker and plot execution times
            for ticker, group in df[df['execution_time'].notna()].groupby('ticker'):
                plt.plot(range(len(group)), group['execution_time'], label=ticker)
                
            plt.title('Execution Time by Ticker')
            plt.xlabel('Execution Sequence')
            plt.ylabel('Execution Time (ms)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/execution_time_{report_time}.png")
            plt.close()
            
        # Generate accuracy visualization
        if 'accuracy' in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Group by ticker and plot accuracies
            for ticker, group in df[df['accuracy'].notna()].groupby('ticker'):
                plt.plot(range(len(group)), group['accuracy'], label=ticker)
                
            plt.title('Prediction Accuracy by Ticker')
            plt.xlabel('Prediction Sequence')
            plt.ylabel('Accuracy (0-1)')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/accuracy_{report_time}.png")
            plt.close()
            
        # Generate optimization events visualization
        if self.optimization_events:
            events_df = pd.DataFrame(self.optimization_events)
            plt.figure(figsize=(12, 6))
            
            reasons = events_df['reason'].unique()
            colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(reasons)]
            reason_color_map = dict(zip(reasons, colors))
            
            # Create scatter plot of optimization events
            for reason, group in events_df.groupby('reason'):
                plt.scatter(
                    range(len(group)), 
                    [1] * len(group),  # All at y=1 for visibility
                    label=reason,
                    s=100,
                    color=reason_color_map[reason],
                    marker='o'
                )
                
                # Add ticker labels
                for i, (_, row) in enumerate(group.iterrows()):
                    plt.annotate(
                        row['ticker'],
                        (i, 1),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center'
                    )
                
            plt.title('Optimization Events Timeline')
            plt.xlabel('Event Sequence')
            plt.yticks([])  # Hide y-axis ticks
            plt.legend()
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/optimization_events_{report_time}.png")
            plt.close()
            
        # Generate summary statistics
        summary = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_duration': str(datetime.now() - self.start_time),
            'tickers_tracked': len(set(df['ticker']) if 'ticker' in df.columns else []),
            'total_executions': len(df[df['execution_time'].notna()]) if 'execution_time' in df.columns else 0,
            'total_predictions': len(df[df['accuracy'].notna()]) if 'accuracy' in df.columns else 0,
            'optimization_candidates': len(self.get_optimization_candidates()),
            'optimization_events': len(self.optimization_events),
            'thresholds': self.optimization_thresholds
        }
        
        # Save summary as JSON
        with open(f"{output_dir}/summary_{report_time}.json", 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Save summary as text
        with open(f"{output_dir}/summary_{report_time}.txt", 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
                
        # Generate HTML report
        self._generate_html_report(df, events_df if self.optimization_events else None, summary, report_time, output_dir)
                
        return f"{output_dir}/performance_report_{report_time}.html"
    
    def _generate_html_report(self, log_df, events_df, summary, report_time, output_dir):
        """
        Generate an HTML performance report
        """
        # Create basic HTML structure
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ticker Skimmer Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .card h3 {{ margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .flex-item {{ flex: 1; min-width: 300px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333366; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .critical {{ color: red; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Ticker Skimmer Performance Report</h1>
            <p>Generated on: {summary['report_time']}</p>
            
            <div class="flex-container">
                <div class="flex-item card">
                    <h3>Run Summary</h3>
                    <p><strong>Duration:</strong> {summary['run_duration']}</p>
                    <p><strong>Tickers Tracked:</strong> <span class="metric-value">{summary['tickers_tracked']}</span></p>
                    <p><strong>Total Executions:</strong> {summary['total_executions']}</p>
                    <p><strong>Total Predictions:</strong> {summary['total_predictions']}</p>
                </div>
                
                <div class="flex-item card">
                    <h3>Optimization Summary</h3>
                    <p><strong>Candidates for Optimization:</strong> <span class="metric-value {self._get_severity_class(summary['optimization_candidates'], 0, 3, 10)}">{summary['optimization_candidates']}</span></p>
                    <p><strong>Optimization Events:</strong> <span class="metric-value {self._get_severity_class(summary['optimization_events'], 0, 5, 15)}">{summary['optimization_events']}</span></p>
                    <p><strong>Execution Time Threshold:</strong> {summary['thresholds']['execution_time']} ms</p>
                    <p><strong>Accuracy Drop Threshold:</strong> {summary['thresholds']['accuracy_drop'] * 100}%</p>
                </div>
            </div>
        """
        
        # Add execution time visualization
        html += f"""
            <h2>Execution Performance</h2>
            <div class="card">
                <h3>Execution Time by Ticker</h3>
                <img src="execution_time_{report_time}.png" alt="Execution Time Chart">
            </div>
        """
        
        # Add optimization candidates table
        candidates = self.get_optimization_candidates()
        if candidates:
            html += f"""
                <div class="card">
                    <h3>Optimization Candidates</h3>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th>Avg Execution Time (ms)</th>
                            <th>Total Executions</th>
                            <th>Recent Performance</th>
                        </tr>
            """
            
            for candidate in candidates:
                recent_times = candidate.get('recent_times', [])
                recent_trend = "↑" if len(recent_times) >= 2 and recent_times[-1] > recent_times[-2] else "↓"
                severity_class = self._get_severity_class(candidate['avg_execution_time'], 50, 100, 200)
                
                html += f"""
                        <tr>
                            <td>{candidate['ticker']}</td>
                            <td class="{severity_class}">{candidate['avg_execution_time']:.2f}</td>
                            <td>{candidate['executions']}</td>
                            <td>{recent_trend} {', '.join([f"{t:.2f}" for t in recent_times])}</td>
                        </tr>
                """
                
            html += """
                    </table>
                </div>
            """
            
        # Add prediction accuracy section
        if 'accuracy' in log_df.columns:
            html += f"""
                <h2>Prediction Performance</h2>
                <div class="card">
                    <h3>Prediction Accuracy by Ticker</h3>
                    <img src="accuracy_{report_time}.png" alt="Accuracy Chart">
                </div>
            """
            
            # Add accuracy stats table
            accuracy_stats = self.get_accuracy_stats()
            if accuracy_stats:
                html += f"""
                    <div class="card">
                        <h3>Accuracy Statistics</h3>
                        <table>
                            <tr>
                                <th>Ticker</th>
                                <th>Mean Accuracy</th>
                                <th>Latest Accuracy</th>
                                <th>Trend</th>
                                <th>Predictions</th>
                            </tr>
                """
                
                for ticker, stats in accuracy_stats.items():
                    mean_class = self._get_accuracy_class(stats['mean_accuracy'])
                    latest_class = self._get_accuracy_class(stats['latest_accuracy'])
                    trend_symbol = "↑" if stats['trend'] == 'improving' else "↓"
                    
                    html += f"""
                            <tr>
                                <td>{ticker}</td>
                                <td class="{mean_class}">{stats['mean_accuracy']:.2f}</td>
                                <td class="{latest_class}">{stats['latest_accuracy']:.2f}</td>
                                <td>{trend_symbol}</td>
                                <td>{stats['num_predictions']}</td>
                            </tr>
                    """
                    
                html += """
                        </table>
                    </div>
                """
                
        # Add optimization events section
        if events_df is not None and not events_df.empty:
            html += f"""
                <h2>Optimization Events</h2>
                <div class="card">
                    <h3>Optimization Events Timeline</h3>
                    <img src="optimization_events_{report_time}.png" alt="Optimization Events Timeline">
                </div>
                
                <div class="card">
                    <h3>Optimization Event Details</h3>
                    <table>
                        <tr>
                            <th>Timestamp</th>
                            <th>Ticker</th>
                            <th>Reason</th>
                            <th>Threshold</th>
                            <th>Current Value</th>
                        </tr>
            """
            
            for _, event in events_df.iterrows():
                html += f"""
                        <tr>
                            <td>{event['timestamp']}</td>
                            <td>{event['ticker']}</td>
                            <td>{event['reason']}</td>
                            <td>{event['threshold']}</td>
                            <td>{event['current_value']:.2f}</td>
                        </tr>
                """
                
            html += """
                    </table>
                </div>
            """
            
        # Close HTML document
        html += """
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f"{output_dir}/performance_report_{report_time}.html", 'w') as f:
            f.write(html)
    
    def _get_severity_class(self, value, good_threshold, warning_threshold, critical_threshold):
        """
        Get CSS class based on severity thresholds
        """
        if value <= good_threshold:
            return "good"
        elif value <= warning_threshold:
            return ""  # Default color
        elif value <= critical_threshold:
            return "warning"
        else:
            return "critical"
            
    def _get_accuracy_class(self, accuracy):
        """
        Get CSS class based on accuracy value
        """
        if accuracy >= 0.8:
            return "good"
        elif accuracy >= 0.6:
            return ""  # Default color
        elif accuracy >= 0.4:
            return "warning"
        else:
            return "critical"
    
    def reset(self):
        """
        Reset the monitor
        """
        self.execution_times = {}
        self.prediction_accuracy = {}
        self.log_data = []
        self.optimization_events = []
        self.start_time = datetime.now()