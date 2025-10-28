#!/usr/bin/env python3
"""
Performance Monitoring and Logging System
=========================================

This module provides comprehensive performance monitoring, logging,
and caching strategies for PicoTuri-EditJudge algorithms.

Features:
- Real-time performance metrics collection
- Algorithm-specific performance tracking
- Memory usage monitoring
- Response time optimization
- Caching strategies for frequently accessed data
- Performance analytics and reporting
"""

import time
import psutil
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from functools import wraps
import json
import gc
import torch


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    algorithm_name: str
    execution_time: float
    memory_usage_mb: float
    gpu_memory_used: float = 0.0
    cpu_percent: float = 0.0
    accuracy_score: Optional[float] = None
    throughput: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None


class PerformanceMonitor:
    """
    Advanced performance monitoring system for algorithms.
    
    Tracks execution time, memory usage, accuracy, and performance metrics
    for all PicoTuri-EditJudge algorithms in real-time.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.algorithm_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.start_times: Dict[str, float] = {}
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            'max_execution_time': 30.0,  # 30 seconds
            'max_memory_mb': 4096,       # 4GB
            'max_gpu_memory_mb': 8192,   # 8GB
            'min_accuracy_score': 0.7    # 70% minimum
        }

    def start_monitoring(self):
        """Start system-wide performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            print("📊 Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join()
        print("📊 Performance monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            self._collect_system_metrics()
            time.sleep(5)  # Monitor every 5 seconds

    def _collect_system_metrics(self):
        """Collect system-wide performance metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics if available
            gpu_memory = 0.0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB

            # Store system metrics
            system_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / 1024**2,
                'gpu_memory_mb': gpu_memory,
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'timestamp': time.time()
            }
            
            self.algorithm_stats['system'] = system_metrics
            
        except Exception as e:
            print(f"⚠️ Error collecting system metrics: {e}")

    def monitor_algorithm(self, algorithm_name: str):
        """Decorator for monitoring algorithm performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                gpu_memory_start = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                
                error_occurred = False
                error_message = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    end_memory = self._get_memory_usage()
                    gpu_memory_end = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                    
                    # Create metrics
                    metrics = PerformanceMetrics(
                        algorithm_name=algorithm_name,
                        execution_time=execution_time,
                        memory_usage_mb=end_memory - start_memory,
                        gpu_memory_used=gpu_memory_end - gpu_memory_start,
                        cpu_percent=psutil.cpu_percent(),
                        error_occurred=error_occurred,
                        error_message=error_message
                    )
                    
                    # Add to history
                    self.metrics_history[algorithm_name].append(metrics)
                    
                    # Check thresholds
                    self._check_performance_thresholds(metrics)
                    
                    # Update algorithm stats
                    self._update_algorithm_stats(algorithm_name, metrics)
                    
            return wrapper
        return decorator

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**2

    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed performance thresholds."""
        warnings = []
        
        if metrics.execution_time > self.thresholds['max_execution_time']:
            warnings.append(f"High execution time: {metrics.execution_time:.2f}s")
            
        if metrics.memory_usage_mb > self.thresholds['max_memory_mb']:
            warnings.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
            
        if metrics.gpu_memory_used > self.thresholds['max_gpu_memory_mb']:
            warnings.append(f"High GPU memory usage: {metrics.gpu_memory_used:.1f}MB")
            
        if warnings:
            print(f"⚠️ Performance warnings for {metrics.algorithm_name}:")
            for warning in warnings:
                print(f"   - {warning}")

    def _update_algorithm_stats(self, algorithm_name: str, metrics: PerformanceMetrics):
        """Update algorithm statistics."""
        if algorithm_name not in self.algorithm_stats:
            self.algorithm_stats[algorithm_name] = {
                'total_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'max_execution_time': 0.0,
                'avg_memory_usage': 0.0,
                'success_rate': 0.0,
                'error_count': 0,
                'recent_performance': deque(maxlen=50)
            }
        
        stats = self.algorithm_stats[algorithm_name]
        stats['total_executions'] += 1
        stats['total_execution_time'] += metrics.execution_time
        stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_executions']
        stats['min_execution_time'] = min(stats['min_execution_time'], metrics.execution_time)
        stats['max_execution_time'] = max(stats['max_execution_time'], metrics.execution_time)
        stats['avg_memory_usage'] = (stats['avg_memory_usage'] * (stats['total_executions'] - 1) + metrics.memory_usage_mb) / stats['total_executions']
        
        if metrics.error_occurred:
            stats['error_count'] += 1
            
        stats['success_rate'] = (stats['total_executions'] - stats['error_count']) / stats['total_executions']
        stats['recent_performance'].append(metrics.execution_time)

    def get_performance_report(self, algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if algorithm_name:
            return self._generate_algorithm_report(algorithm_name)
        else:
            return self._generate_system_report()

    def _generate_algorithm_report(self, algorithm_name: str) -> Dict[str, Any]:
        """Generate performance report for specific algorithm."""
        if algorithm_name not in self.algorithm_stats:
            return {"error": f"No data found for algorithm: {algorithm_name}"}
        
        stats = self.algorithm_stats[algorithm_name]
        recent_executions = list(self.metrics_history[algorithm_name])[-10:]  # Last 10 executions
        
        return {
            'algorithm_name': algorithm_name,
            'total_executions': stats['total_executions'],
            'performance_metrics': {
                'avg_execution_time': f"{stats['avg_execution_time']:.3f}s",
                'min_execution_time': f"{stats['min_execution_time']:.3f}s",
                'max_execution_time': f"{stats['max_execution_time']:.3f}s",
                'avg_memory_usage': f"{stats['avg_memory_usage']:.1f}MB",
                'success_rate': f"{stats['success_rate']:.1%}",
                'error_count': stats['error_count']
            },
            'recent_performance': {
                'executions': len(recent_executions),
                'avg_time': f"{sum(e.execution_time for e in recent_executions) / len(recent_executions):.3f}s" if recent_executions else "0s",
                'avg_memory': f"{sum(e.memory_usage_mb for e in recent_executions) / len(recent_executions):.1f}MB" if recent_executions else "0MB"
            },
            'performance_trend': 'improving' if len(recent_executions) >= 2 and recent_executions[-1].execution_time < recent_executions[0].execution_time else 'stable'
        }

    def _generate_system_report(self) -> Dict[str, Any]:
        """Generate system-wide performance report."""
        reports = {}
        for algo_name in self.algorithm_stats:
            if algo_name != 'system':
                reports[algo_name] = self._generate_algorithm_report(algo_name)
        
        system_info = self.algorithm_stats.get('system', {})
        
        return {
            'system_performance': system_info,
            'algorithm_reports': reports,
            'summary': {
                'total_algorithms_monitored': len([k for k in self.algorithm_stats.keys() if k != 'system']),
                'active_monitoring': self.monitoring_active,
                'monitoring_duration': time.time() - getattr(self, '_start_time', time.time())
            }
        }

    def save_performance_log(self, filepath: str = "performance_log.json"):
        """Save performance metrics to JSON file."""
        data = {
            'timestamp': time.time(),
            'algorithm_stats': dict(self.algorithm_stats),
            'metrics_history': {k: [self._metrics_to_dict(m) for m in v] for k, v in self.metrics_history.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"📊 Performance log saved to {filepath}")

    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert PerformanceMetrics to dictionary."""
        return {
            'algorithm_name': metrics.algorithm_name,
            'execution_time': metrics.execution_time,
            'memory_usage_mb': metrics.memory_usage_mb,
            'gpu_memory_used': metrics.gpu_memory_used,
            'cpu_percent': metrics.cpu_percent,
            'accuracy_score': metrics.accuracy_score,
            'throughput': metrics.throughput,
            'error_occurred': metrics.error_occurred,
            'error_message': metrics.error_message
        }


class OptimizedCache:
    """
    High-performance caching system for algorithm results.
    
    Implements multiple cache strategies:
    - LRU (Least Recently Used)
    - TTL (Time To Live)
    - Memory-based eviction
    - Compressed storage for large objects
    """

    def __init__(self, max_size_mb: int = 512, default_ttl: int = 3600):
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        self.total_size = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL checking."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry['timestamp'] > entry.get('ttl', self.default_ttl):
                self._remove_entry(key)
                return None
            
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, compress: bool = False) -> None:
        """Set value in cache with optional compression."""
        with self._lock:
            import pickle
            import zlib
            
            # Serialize and optionally compress
            if compress and hasattr(value, '__dict__'):
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized)
                value = {
                    'data': compressed,
                    'compressed': True,
                    'original_size': len(serialized),
                    'compressed_size': len(compressed)
                }
            
            value_size = self._calculate_size(value)
            ttl = ttl or self.default_ttl
            
            # Check if we need to evict entries
            while self.total_size + value_size > self.max_size_mb * 1024 * 1024 and self.cache:
                self._evict_lru()
            
            # Remove old entry if exists
            if key in self.cache:
                self.total_size -= self._calculate_size(self.cache[key])
            
            # Add new entry
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl,
                'size': value_size
            }
            
            if key not in self.access_order:
                self.access_order.append(key)
            
            self.total_size += value_size
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes."""
        import sys
        try:
            return sys.getsizeof(obj)
        except:
            return 1024  # Default size estimate
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            self.total_size -= self.cache[key]['size']
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove_entry(lru_key)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'total_entries': len(self.cache),
                'total_size_mb': self.total_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'utilization': (self.total_size / (self.max_size_mb * 1024 * 1024)) * 100,
                'access_order_length': len(self.access_order)
            }


class PerformanceOptimizer:
    """
    Performance optimization utilities for algorithms.
    
    Provides memory optimization, garbage collection, and performance tuning.
    """

    @staticmethod
    def optimize_memory():
        """Perform memory optimization."""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_cleared = torch.cuda.memory_reserved() / 1024**2
        else:
            gpu_cleared = 0
        
        return {
            'garbage_collected': collected,
            'gpu_cache_cleared_mb': gpu_cleared,
            'memory_freed_mb': collected * 0.001  # Rough estimate
        }

    @staticmethod
    def batch_process(items: List[Any], process_func: Callable, batch_size: int = 32) -> List[Any]:
        """Optimized batch processing with memory management."""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
            
            # Periodic memory optimization
            if i % (batch_size * 10) == 0:
                PerformanceOptimizer.optimize_memory()
        
        return results

    @staticmethod
    def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile function execution with detailed metrics."""
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2
        
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
        except Exception as e:
            profiler.disable()
            raise e
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024**2
        
        # Get profiling stats
        stats_io = StringIO()
        stats = pstats.Stats(profiler, stream=stats_io)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        return {
            'execution_time': end_time - start_time,
            'memory_used_mb': end_memory - start_memory,
            'return_value': result,
            'top_functions': stats_io.getvalue()
        }


# Global instances
performance_monitor = PerformanceMonitor()
result_cache = OptimizedCache(max_size_mb=256)


def initialize_monitoring():
    """Initialize performance monitoring system."""
    performance_monitor.start_monitoring()
    print("🚀 Performance monitoring and caching initialized")


def get_cached_result(cache_key: str, compute_func: Callable, *args, **kwargs) -> Any:
    """Get cached result or compute and cache new result."""
    # Try to get from cache first
    cached_result = result_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Compute new result
    result = compute_func(*args, **kwargs)
    
    # Cache the result
    result_cache.set(cache_key, result)
    
    return result


# Demo function
def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("📊 Performance Monitoring System Demo")
    print("=" * 40)
    
    # Initialize monitoring
    initialize_monitoring()
    
    # Test memory optimization
    print("\n🧹 Testing memory optimization...")
    opt_result = PerformanceOptimizer.optimize_memory()
    print(f"   Garbage collected: {opt_result['garbage_collected']} objects")
    print(f"   GPU cache cleared: {opt_result['gpu_cache_cleared_mb']:.1f}MB")
    
    # Test caching
    print("\n💾 Testing caching system...")
    
    def expensive_computation(x):
        time.sleep(0.1)  # Simulate expensive operation
        return x ** 2
    
    # First call (will compute)
    start_time = time.time()
    result1 = get_cached_result("test_key", expensive_computation, 42)
    first_call_time = time.time() - start_time
    
    # Second call (will use cache)
    start_time = time.time()
    result2 = get_cached_result("test_key", expensive_computation, 42)
    second_call_time = time.time() - start_time
    
    print(f"   First call time: {first_call_time:.3f}s")
    print(f"   Second call time: {second_call_time:.3f}s")
    print(f"   Speedup: {first_call_time / second_call_time:.1f}x")
    print(f"   Results match: {result1 == result2}")
    
    # Show cache statistics
    cache_stats = result_cache.get_stats()
    print(f"\n📈 Cache Statistics:")
    print(f"   Total entries: {cache_stats['total_entries']}")
    print(f"   Cache utilization: {cache_stats['utilization']:.1f}%")
    
    # Generate performance report
    print("\n📋 Performance Report:")
    report = performance_monitor.get_performance_report()
    print(json.dumps(report, indent=2))
    
    print("\n🎯 Performance Monitoring Status: ACTIVE ✅")
    print("📊 Real-time metrics collection enabled")
    print("💾 Caching and optimization active")


if __name__ == "__main__":
    demo_performance_monitoring()
