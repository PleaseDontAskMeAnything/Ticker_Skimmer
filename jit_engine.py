"""
JIT Engine for Ticker Skimmer
Core optimization engine that provides JIT-like functionality for ticker analysis
"""
import time
import numpy as np
import pandas as pd
from functools import lru_cache

class JITEngine:
    """
    JIT Engine provides optimization for frequently executed code paths
    in ticker analysis by dynamically optimizing hot paths.
    """
    def __init__(self):
        self.compilation_cache = {}
        self.execution_stats = {}
        self.hot_threshold = 10  # Number of executions before optimization
        
    def should_optimize(self, key):
        """
        Determine if a code path should be optimized
        """
        if key not in self.execution_stats:
            self.execution_stats[key] = {
                'executions': 0,
                'total_time': 0,
                'optimized': False
            }
            
        return (self.execution_stats[key]['executions'] >= self.hot_threshold and 
                not self.execution_stats[key]['optimized'])
    
    def track_execution(self, key, execution_time):
        """
        Track execution statistics for a code path
        """
        if key not in self.execution_stats:
            self.execution_stats[key] = {
                'executions': 0,
                'total_time': 0,
                'optimized': False
            }
            
        self.execution_stats[key]['executions'] += 1
        self.execution_stats[key]['total_time'] += execution_time
        
    def optimize(self, key, optimization_func):
        """
        Optimize a code path by caching its optimized version
        """
        self.compilation_cache[key] = optimization_func()
        self.execution_stats[key]['optimized'] = True
        
    def get_optimized_func(self, key):
        """
        Get the optimized function for a key if available
        """
        return self.compilation_cache.get(key)
    
    def get_stats(self):
        """
        Get execution statistics for all tracked code paths
        """
        return self.execution_stats
    
    def reset_stats(self):
        """
        Reset execution statistics
        """
        self.execution_stats = {}
        
    def clear_cache(self):
        """
        Clear the compilation cache
        """
        self.compilation_cache = {}

# Decorators for simplifying JIT usage
def jit_optimized(engine, key_func=None):
    """
    Decorator to apply JIT optimization to a function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate a unique key for this function call
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default to function name if no key function provided
                key = func.__name__
                
            # Check if we have an optimized version
            optimized_func = engine.get_optimized_func(key)
            if optimized_func:
                return optimized_func(*args, **kwargs)
            
            # Track execution time
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Track this execution
            engine.track_execution(key, execution_time)
            
            # Check if we should optimize
            if engine.should_optimize(key):
                # Create an optimized version (could be a memoized version, etc.)
                def optimized_version():
                    # This is where you'd put more complex optimization logic
                    return lru_cache(maxsize=128)(func)
                
                engine.optimize(key, optimized_version)
                
            return result
        return wrapper
    return decorator

# Utility functions for common optimizations
def optimize_dataframe_operations(df, operations):
    """
    Optimize a series of operations on a DataFrame
    """
    # Cache intermediate results for reuse
    results_cache = {}
    final_results = {}
    
    for op_name, op_func, op_args in operations:
        if op_name in results_cache:
            # Reuse cached result
            result = results_cache[op_name]
        else:
            # Execute operation and cache result
            result = op_func(df, *op_args)
            results_cache[op_name] = result
            
        final_results[op_name] = result
        
    return final_results