"""
Adaptive Micro-Batcher
Dynamic batch sizing for optimal throughput and latency in PicoTuri-EditJudge
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class BatchStrategy(Enum):
    """Batching strategies for different optimization goals"""
    THROUGHPUT = "throughput"  # Maximize throughput, higher latency
    LATENCY = "latency"        # Minimize latency, lower throughput
    BALANCED = "balanced"      # Balance between throughput and latency
    ADAPTIVE = "adaptive"      # Dynamically adjust based on load

@dataclass
class BatchMetrics:
    """Metrics for batch performance monitoring"""
    batch_size: int
    processing_time: float
    throughput: float  # samples per second
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_usage: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    
    def __post_init__(self):
        if not self.future.done():
            self.future.set_result(None)

class AdaptiveMicroBatcher:
    """
    Adaptive micro-batcher that dynamically adjusts batch size based on:
    - Current load and queue length
    - Historical performance metrics
    - Latency requirements
    - Memory constraints
    """
    
    def __init__(
        self,
        strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        target_latency_ms: float = 50.0,
        max_wait_time_ms: float = 10.0,
        memory_limit_mb: float = 2048.0,
        metrics_window: int = 100,
        adjustment_factor: float = 0.1,
        enable_monitoring: bool = True
    ):
        """
        Initialize adaptive micro-batcher
        
        Args:
            strategy: Batching strategy to use
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_latency_ms: Target latency in milliseconds
            max_wait_time_ms: Maximum wait time before processing partial batch
            memory_limit_mb: Memory limit in MB
            metrics_window: Number of recent metrics to consider for adaptation
            adjustment_factor: Factor for batch size adjustment
            enable_monitoring: Enable performance monitoring
        """
        self.strategy = strategy
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.max_wait_time_ms = max_wait_time_ms
        self.memory_limit_mb = memory_limit_mb
        self.metrics_window = metrics_window
        self.adjustment_factor = adjustment_factor
        self.enable_monitoring = enable_monitoring
        
        # Current state
        self.current_batch_size = min_batch_size
        self.request_queue = deque()
        self.processing_lock = asyncio.Lock()
        self.is_running = False
        
        # Performance tracking
        self.metrics_history = deque(maxlen=metrics_window)
        self.total_requests_processed = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
        # Adaptive parameters
        self.last_adjustment_time = time.time()
        self.adjustment_interval = 5.0  # Adjust every 5 seconds
        self.load_threshold = 0.8  # Queue utilization threshold
        
        # Thread pool for CPU-bound processing
        self.executor = ThreadPoolExecutor(
            max_workers=min(8, max_batch_size),
            thread_name_prefix="batcher"
        )
        
        logger.info(f"AdaptiveMicroBatcher initialized with strategy: {strategy.value}")
    
    async def start(self, process_func: Callable[[List[BatchRequest]], List[Any]]):
        """
        Start the batcher with a processing function
        
        Args:
            process_func: Async function that processes a batch of requests
        """
        if self.is_running:
            logger.warning("Batcher is already running")
            return
        
        self.is_running = True
        self.process_func = process_func
        
        logger.info("Starting adaptive micro-batcher")
        
        # Start the main processing loop
        await self._processing_loop()
    
    async def stop(self):
        """Stop the batcher gracefully"""
        logger.info("Stopping adaptive micro-batcher")
        
        self.is_running = False
        
        # Process remaining requests
        if self.request_queue:
            logger.info(f"Processing {len(self.request_queue)} remaining requests")
            await self._process_batch(list(self.request_queue))
            self.request_queue.clear()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Batcher stopped")
    
    async def submit_request(
        self,
        request_id: str,
        data: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """
        Submit a request for batched processing
        
        Args:
            request_id: Unique identifier for the request
            data: Request data
            timeout: Optional timeout for request completion
            
        Returns:
            Processing result
        """
        if not self.is_running:
            raise RuntimeError("Batcher is not running")
        
        # Create request
        request = BatchRequest(request_id=request_id, data=data)
        
        # Add to queue
        self.request_queue.append(request)
        
        # Wait for result
        try:
            if timeout:
                result = await asyncio.wait_for(request.future, timeout=timeout)
            else:
                result = await request.future
            return result
        except asyncio.TimeoutError:
            # Remove from queue if still there
            try:
                self.request_queue.remove(request)
            except ValueError:
                pass
            raise TimeoutError(f"Request {request_id} timed out")
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Determine optimal batch size
                batch_size = self._get_optimal_batch_size()
                
                # Wait for batch or timeout
                batch = await self._wait_for_batch(batch_size)
                
                if batch:
                    # Process the batch
                    await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _wait_for_batch(self, target_size: int) -> List[BatchRequest]:
        """
        Wait for requests to form a batch
        
        Args:
            target_size: Target batch size
            
        Returns:
            List of requests to process
        """
        start_time = time.time()
        max_wait_time = self.max_wait_time_ms / 1000.0
        
        while len(self.request_queue) < target_size and self.is_running:
            # Check if we've waited too long
            if time.time() - start_time > max_wait_time:
                # Process whatever we have
                break
            
            # Check if we have enough for a smaller batch
            if len(self.request_queue) >= self.min_batch_size:
                # Under high load, process smaller batches more frequently
                queue_utilization = len(self.request_queue) / self.max_batch_size
                if queue_utilization > self.load_threshold:
                    break
            
            await asyncio.sleep(0.001)  # 1ms poll interval
        
        # Extract batch
        batch_size = min(len(self.request_queue), target_size)
        batch = [self.request_queue.popleft() for _ in range(batch_size)]
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """
        Process a batch of requests
        
        Args:
            batch: List of requests to process
        """
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Extract data for processing
            batch_data = [req.data for req in batch]
            
            # Process batch
            results = await self.process_func(batch_data)
            
            # Set results for each request
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_metrics(batch, processing_time)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Set exception for all requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    def _get_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on current conditions
        
        Returns:
            Optimal batch size
        """
        if self.strategy == BatchStrategy.THROUGHPUT:
            return self.max_batch_size
        
        elif self.strategy == BatchStrategy.LATENCY:
            return self.min_batch_size
        
        elif self.strategy == BatchStrategy.BALANCED:
            return (self.min_batch_size + self.max_batch_size) // 2
        
        elif self.strategy == BatchStrategy.ADAPTIVE:
            return self._calculate_adaptive_batch_size()
        
        return self.current_batch_size
    
    def _calculate_adaptive_batch_size(self) -> int:
        """
        Calculate adaptive batch size based on performance metrics and load
        
        Returns:
            Adaptive batch size
        """
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return self.current_batch_size
        
        # Get current load
        queue_length = len(self.request_queue)
        queue_utilization = queue_length / self.max_batch_size
        
        # Get recent performance metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        if not recent_metrics:
            # No metrics yet, start with minimum
            return self.min_batch_size
        
        # Calculate average latency and throughput
        avg_latency = np.mean([m.processing_time for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        
        # Determine adjustment direction
        new_batch_size = self.current_batch_size
        
        # High load - increase batch size
        if queue_utilization > self.load_threshold:
            if avg_latency < self.target_latency_ms:
                # Can increase batch size
                new_batch_size = min(
                    self.current_batch_size + max(1, int(self.current_batch_size * self.adjustment_factor)),
                    self.max_batch_size
                )
        
        # Low load or high latency - decrease batch size
        elif queue_utilization < 0.3 or avg_latency > self.target_latency_ms * 1.2:
            new_batch_size = max(
                self.current_batch_size - max(1, int(self.current_batch_size * self.adjustment_factor)),
                self.min_batch_size
            )
        
        # Check memory constraints
        estimated_memory = self._estimate_memory_usage(new_batch_size)
        if estimated_memory > self.memory_limit_mb:
            new_batch_size = max(
                int(new_batch_size * (self.memory_limit_mb / estimated_memory)),
                self.min_batch_size
            )
        
        # Update current batch size if changed
        if new_batch_size != self.current_batch_size:
            logger.info(
                f"Adjusting batch size: {self.current_batch_size} -> {new_batch_size} "
                f"(load: {queue_utilization:.2f}, latency: {avg_latency:.1f}ms)"
            )
            self.current_batch_size = new_batch_size
            self.last_adjustment_time = current_time
        
        return self.current_batch_size
    
    def _estimate_memory_usage(self, batch_size: int) -> float:
        """
        Estimate memory usage for a given batch size
        
        Args:
            batch_size: Batch size to estimate for
            
        Returns:
            Estimated memory usage in MB
        """
        # Base memory + per-sample memory
        base_memory_mb = 100.0  # Base model memory
        per_sample_mb = 2.0     # Memory per sample (embeddings + intermediate)
        
        return base_memory_mb + (batch_size * per_sample_mb)
    
    def _update_metrics(self, batch: List[BatchRequest], processing_time: float):
        """
        Update performance metrics
        
        Args:
            batch: Processed batch
            processing_time: Processing time in milliseconds
        """
        batch_size = len(batch)
        throughput = (batch_size / processing_time) * 1000  # samples per second
        
        # Calculate latencies for individual requests
        latencies = []
        for req in batch:
            latency = (time.time() - req.timestamp) * 1000  # ms
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        # Create metrics
        metrics = BatchMetrics(
            batch_size=batch_size,
            processing_time=processing_time,
            throughput=throughput,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            memory_usage=self._estimate_memory_usage(batch_size)
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.total_requests_processed += batch_size
        self.total_processing_time += processing_time
        
        # Log if monitoring enabled
        if self.enable_monitoring and len(self.metrics_history) % 10 == 0:
            self._log_performance_summary()
    
    def _log_performance_summary(self):
        """Log performance summary"""
        if not self.metrics_history:
            return
        
        recent = list(self.metrics_history)[-10:]
        avg_throughput = np.mean([m.throughput for m in recent])
        avg_latency = np.mean([m.processing_time for m in recent])
        p95_latency = np.mean([m.latency_p95 for m in recent])
        
        logger.info(
            f"Performance Summary - Batch Size: {self.current_batch_size}, "
            f"Avg Throughput: {avg_throughput:.1f} samples/s, "
            f"Avg Latency: {avg_latency:.1f}ms, "
            f"P95 Latency: {p95_latency:.1f}ms, "
            f"Queue Length: {len(self.request_queue)}"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.metrics_history:
            return {}
        
        recent = list(self.metrics_history)[-10:]
        
        return {
            'current_batch_size': self.current_batch_size,
            'queue_length': len(self.request_queue),
            'total_requests_processed': self.total_requests_processed,
            'avg_throughput': np.mean([m.throughput for m in recent]),
            'avg_latency_ms': np.mean([m.processing_time for m in recent]),
            'p95_latency_ms': np.mean([m.latency_p95 for m in recent]),
            'p99_latency_ms': np.mean([m.latency_p99 for m in recent]),
            'memory_usage_mb': self._estimate_memory_usage(self.current_batch_size),
            'uptime_seconds': time.time() - self.start_time,
            'strategy': self.strategy.value
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status
        
        Returns:
            Queue status information
        """
        return {
            'queue_length': len(self.request_queue),
            'max_queue_size': self.max_batch_size,
            'queue_utilization': len(self.request_queue) / self.max_batch_size,
            'oldest_request_age': (
                time.time() - self.request_queue[0].timestamp
                if self.request_queue else 0
            )
        }

# Test function
async def test_adaptive_batcher():
    """Test the adaptive micro-batcher"""
    print("Testing Adaptive Micro-Batcher...")
    
    # Mock processing function
    async def mock_process_batch(batch_data):
        # Simulate processing time
        await asyncio.sleep(0.01)  # 10ms processing time
        return [f"result_{i}" for i in range(len(batch_data))]
    
    # Create batcher
    batcher = AdaptiveMicroBatcher(
        strategy=BatchStrategy.ADAPTIVE,
        min_batch_size=1,
        max_batch_size=16,
        target_latency_ms=50.0,
        max_wait_time_ms=20.0
    )
    
    # Start batcher in background
    batcher_task = asyncio.create_task(batcher.start(mock_process_batch))
    
    try:
        # Submit some test requests
        tasks = []
        for i in range(50):
            task = asyncio.create_task(
                batcher.submit_request(
                    request_id=f"req_{i}",
                    data={"text": f"test request {i}"}
                )
            )
            tasks.append(task)
            
            # Add some delay between requests
            await asyncio.sleep(0.001)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        
        print(f"Processed {len(results)} requests")
        
        # Get metrics
        metrics = batcher.get_metrics()
        print(f"Metrics: {metrics}")
        
        # Get queue status
        queue_status = batcher.get_queue_status()
        print(f"Queue status: {queue_status}")
        
    finally:
        # Stop batcher
        await batcher.stop()
        batcher_task.cancel()
    
    print("Adaptive Micro-Batcher test completed!")

if __name__ == "__main__":
    asyncio.run(test_adaptive_batcher())
