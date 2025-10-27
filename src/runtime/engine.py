"""
Runtime Engine
Session pool and orchestration for PicoTuri-EditJudge
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import onnxruntime as ort
from contextlib import asynccontextmanager
import weakref
from enum import Enum

from .batcher import AdaptiveMicroBatcher, BatchStrategy

logger = logging.getLogger(__name__)

class SessionState(Enum):
    """Session states"""
    IDLE = "idle"
    BUSY = "busy"
    WARMING_UP = "warming_up"
    ERROR = "error"

@dataclass
class SessionInfo:
    """Information about a model session"""
    session_id: str
    model_type: str  # 'text_encoder', 'image_encoder', 'fusion_head'
    session: ort.InferenceSession
    state: SessionState = SessionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    total_inference_time: float = 0.0
    memory_usage: float = 0.0
    
    def update_usage(self, inference_time: float):
        """Update session usage statistics"""
        self.last_used = time.time()
        self.usage_count += 1
        self.total_inference_time += inference_time

@dataclass
class InferenceRequest:
    """Inference request data"""
    request_id: str
    model_type: str
    inputs: Dict[str, np.ndarray]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher priority = processed first

@dataclass
class InferenceResult:
    """Inference result data"""
    request_id: str
    model_type: str
    outputs: Dict[str, np.ndarray]
    processing_time: float
    session_id: str
    timestamp: float = field(default_factory=time.time)

class SessionPool:
    """
    Pool of ONNX Runtime sessions for efficient model inference
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        pool_size: int = 4,
        session_options: Optional[ort.SessionOptions] = None,
        warm_up: bool = True,
        max_idle_time: float = 300.0,  # 5 minutes
        enable_monitoring: bool = True
    ):
        """
        Initialize session pool
        
        Args:
            model_paths: Dictionary mapping model types to file paths
            pool_size: Number of sessions per model type
            session_options: ONNX Runtime session options
            warm_up: Whether to warm up sessions
            max_idle_time: Maximum idle time before session cleanup
            enable_monitoring: Enable performance monitoring
        """
        self.model_paths = model_paths
        self.pool_size = pool_size
        self.session_options = session_options or self._create_default_session_options()
        self.warm_up = warm_up
        self.max_idle_time = max_idle_time
        self.enable_monitoring = enable_monitoring
        
        # Session management
        self.sessions: Dict[str, List[SessionInfo]] = {}
        self.available_sessions: Dict[str, asyncio.Queue] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        
        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"SessionPool initialized with {len(model_paths)} model types")
    
    def _create_default_session_options(self) -> ort.SessionOptions:
        """Create default ONNX Runtime session options"""
        options = ort.SessionOptions()
        
        # Performance optimizations
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Set number of intra-op threads
        options.intra_op_num_threads = min(4, self.pool_size)
        
        return options
    
    async def initialize(self):
        """Initialize all sessions in the pool"""
        logger.info("Initializing session pool...")
        
        for model_type, model_path in self.model_paths.items():
            await self._initialize_model_sessions(model_type, model_path)
        
        # Start cleanup task
        if self.max_idle_time > 0:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Session pool initialized successfully")
    
    async def _initialize_model_sessions(self, model_type: str, model_path: str):
        """Initialize sessions for a specific model type"""
        logger.info(f"Initializing {self.pool_size} sessions for {model_type}")
        
        self.sessions[model_type] = []
        self.available_sessions[model_type] = asyncio.Queue(maxsize=self.pool_size)
        self.session_locks[model_type] = asyncio.Lock()
        
        for i in range(self.pool_size):
            session_id = f"{model_type}_{i}"
            
            try:
                # Create session in thread pool to avoid blocking
                session = await asyncio.get_event_loop().run_in_executor(
                    None, self._create_session, model_path
                )
                
                session_info = SessionInfo(
                    session_id=session_id,
                    model_type=model_type,
                    session=session
                )
                
                self.sessions[model_type].append(session_info)
                self.available_sessions[model_type].put_nowait(session_info)
                
                logger.debug(f"Created session {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to create session {session_id}: {e}")
                raise
        
        # Warm up sessions if enabled
        if self.warm_up:
            await self._warm_up_sessions(model_type)
    
    def _create_session(self, model_path: str) -> ort.InferenceSession:
        """Create ONNX Runtime session"""
        return ort.InferenceSession(model_path, self.session_options)
    
    async def _warm_up_sessions(self, model_type: str):
        """Warm up sessions for a model type"""
        logger.info(f"Warming up sessions for {model_type}")
        
        # Get a session for warm-up
        session_info = await self.get_session(model_type)
        
        try:
            # Create dummy inputs based on model type
            dummy_inputs = self._create_dummy_inputs(model_type)
            
            # Run inference
            start_time = time.time()
            session_info.session.run(None, dummy_inputs)
            inference_time = time.time() - start_time
            
            session_info.update_usage(inference_time)
            
            logger.debug(f"Warmed up session {session_info.session_id}")
            
        except Exception as e:
            logger.warning(f"Failed to warm up session {session_info.session_id}: {e}")
        finally:
            await self.return_session(session_info)
    
    def _create_dummy_inputs(self, model_type: str) -> Dict[str, np.ndarray]:
        """Create dummy inputs for warm-up"""
        if model_type == "text_encoder":
            return {
                "input_ids": np.random.randint(0, 30000, (1, 512), dtype=np.int64),
                "attention_mask": np.ones((1, 512), dtype=np.int64)
            }
        elif model_type == "image_encoder":
            return {
                "image": np.random.random((1, 3, 224, 224)).astype(np.float32)
            }
        elif model_type == "fusion_head":
            return {
                "features": np.random.random((1, 1280)).astype(np.float32)
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @asynccontextmanager
    async def get_session_context(self, model_type: str):
        """Context manager for getting and returning a session"""
        session_info = await self.get_session(model_type)
        try:
            yield session_info
        finally:
            await self.return_session(session_info)
    
    async def get_session(self, model_type: str) -> SessionInfo:
        """
        Get an available session for a model type
        
        Args:
            model_type: Type of model to get session for
            
        Returns:
            Available session info
        """
        if model_type not in self.available_sessions:
            raise ValueError(f"No sessions available for model type: {model_type}")
        
        # Wait for available session
        session_info = await self.available_sessions[model_type].get()
        session_info.state = SessionState.BUSY
        
        return session_info
    
    async def return_session(self, session_info: SessionInfo):
        """
        Return a session to the pool
        
        Args:
            session_info: Session info to return
        """
        if session_info.state != SessionState.ERROR:
            session_info.state = SessionState.IDLE
            await self.available_sessions[session_info.model_type].put(session_info)
    
    async def run_inference(
        self,
        request: InferenceRequest
    ) -> InferenceResult:
        """
        Run inference on a request
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        async with self.get_session_context(request.model_type) as session_info:
            try:
                # Run inference
                outputs = session_info.session.run(None, request.inputs)
                
                # Convert outputs to dictionary
                output_names = session_info.session.get_outputs()
                output_dict = {
                    output.name: output for output, output in zip(output_names, outputs)
                }
                
                processing_time = time.time() - start_time
                
                # Update session usage
                session_info.update_usage(processing_time)
                
                # Update statistics
                self.total_requests += 1
                self.total_inference_time += processing_time
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_type=request.model_type,
                    outputs=output_dict,
                    processing_time=processing_time,
                    session_id=session_info.session_id
                )
                
            except Exception as e:
                session_info.state = SessionState.ERROR
                logger.error(f"Inference failed for request {request.request_id}: {e}")
                raise
    
    async def _cleanup_loop(self):
        """Background cleanup task for idle sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_idle_sessions(self):
        """Clean up idle sessions"""
        current_time = time.time()
        
        for model_type, sessions in self.sessions.items():
            for session_info in sessions:
                idle_time = current_time - session_info.last_used
                
                if idle_time > self.max_idle_time and session_info.state == SessionState.IDLE:
                    logger.info(f"Cleaning up idle session {session_info.session_id}")
                    
                    # Remove from pool
                    sessions.remove(session_info)
                    
                    # Create replacement session
                    try:
                        new_session = await asyncio.get_event_loop().run_in_executor(
                            None, self._create_session, self.model_paths[model_type]
                        )
                        
                        new_session_info = SessionInfo(
                            session_id=session_info.session_id,
                            model_type=model_type,
                            session=new_session
                        )
                        
                        sessions.append(new_session_info)
                        await self.available_sessions[model_type].put(new_session_info)
                        
                    except Exception as e:
                        logger.error(f"Failed to recreate session {session_info.session_id}: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = {
            'total_requests': self.total_requests,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': (
                self.total_inference_time / self.total_requests
                if self.total_requests > 0 else 0
            ),
            'uptime': time.time() - self.start_time,
            'model_types': {}
        }
        
        for model_type, sessions in self.sessions.items():
            available_count = self.available_sessions[model_type].qsize()
            busy_count = sum(1 for s in sessions if s.state == SessionState.BUSY)
            
            stats['model_types'][model_type] = {
                'total_sessions': len(sessions),
                'available_sessions': available_count,
                'busy_sessions': busy_count,
                'total_usage': sum(s.usage_count for s in sessions),
                'avg_usage': np.mean([s.usage_count for s in sessions]) if sessions else 0
            }
        
        return stats
    
    async def shutdown(self):
        """Shutdown the session pool"""
        logger.info("Shutting down session pool...")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        for model_type, sessions in self.sessions.items():
            for session_info in sessions:
                try:
                    session_info.session.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing session {session_info.session_id}: {e}")
        
        logger.info("Session pool shutdown complete")

class RuntimeEngine:
    """
    Main runtime engine that orchestrates batch processing and session management
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        pool_size: int = 4,
        batch_strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
        enable_batching: bool = True,
        **kwargs
    ):
        """
        Initialize runtime engine
        
        Args:
            model_paths: Dictionary mapping model types to file paths
            pool_size: Number of sessions per model type
            batch_strategy: Batching strategy to use
            enable_batching: Whether to enable batch processing
            **kwargs: Additional arguments for session pool and batcher
        """
        self.model_paths = model_paths
        self.pool_size = pool_size
        self.batch_strategy = batch_strategy
        self.enable_batching = enable_batching
        
        # Initialize components
        self.session_pool = SessionPool(
            model_paths=model_paths,
            pool_size=pool_size,
            **kwargs
        )
        
        self.batcher = AdaptiveMicroBatcher(
            strategy=batch_strategy,
            **kwargs
        ) if enable_batching else None
        
        self.is_initialized = False
        self.is_running = False
        
        logger.info("RuntimeEngine initialized")
    
    async def initialize(self):
        """Initialize the runtime engine"""
        if self.is_initialized:
            return
        
        logger.info("Initializing runtime engine...")
        
        # Initialize session pool
        await self.session_pool.initialize()
        
        # Initialize batcher if enabled
        if self.batcher:
            await self.batcher.start(self._process_batch)
        
        self.is_initialized = True
        logger.info("Runtime engine initialized successfully")
    
    async def shutdown(self):
        """Shutdown the runtime engine"""
        logger.info("Shutting down runtime engine...")
        
        # Stop batcher
        if self.batcher:
            await self.batcher.stop()
        
        # Shutdown session pool
        await self.session_pool.shutdown()
        
        self.is_running = False
        self.is_initialized = False
        
        logger.info("Runtime engine shutdown complete")
    
    async def run_text_encoder(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        request_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Run text encoder inference
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            request_id: Optional request ID
            
        Returns:
            Text embeddings
        """
        request_id = request_id or f"text_{int(time.time() * 1000000)}"
        
        request = InferenceRequest(
            request_id=request_id,
            model_type="text_encoder",
            inputs={
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        
        if self.enable_batching and self.batcher:
            # Submit to batcher
            result = await self.batcher.submit_request(request_id, request.__dict__)
            return result["last_hidden_state"]
        else:
            # Direct inference
            result = await self.session_pool.run_inference(request)
            return result.outputs["last_hidden_state"]
    
    async def run_image_encoder(
        self,
        image: np.ndarray,
        request_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Run image encoder inference
        
        Args:
            image: Input image tensor
            request_id: Optional request ID
            
        Returns:
            Image embeddings
        """
        request_id = request_id or f"image_{int(time.time() * 1000000)}"
        
        request = InferenceRequest(
            request_id=request_id,
            model_type="image_encoder",
            inputs={"image": image}
        )
        
        if self.enable_batching and self.batcher:
            # Submit to batcher
            result = await self.batcher.submit_request(request_id, request.__dict__)
            return result["image_features"]
        else:
            # Direct inference
            result = await self.session_pool.run_inference(request)
            return result.outputs["image_features"]
    
    async def run_fusion_head(
        self,
        features: np.ndarray,
        request_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Run fusion head inference
        
        Args:
            features: Fused features
            request_id: Optional request ID
            
        Returns:
            Quality scores
        """
        request_id = request_id or f"fusion_{int(time.time() * 1000000)}"
        
        request = InferenceRequest(
            request_id=request_id,
            model_type="fusion_head",
            inputs={"features": features}
        )
        
        if self.enable_batching and self.batcher:
            # Submit to batcher
            result = await self.batcher.submit_request(request_id, request.__dict__)
            return result["scores"]
        else:
            # Direct inference
            result = await self.session_pool.run_inference(request)
            return result.outputs["scores"]
    
    async def _process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of inference requests
        
        Args:
            batch_data: List of request dictionaries
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # Group requests by model type for efficient processing
        requests_by_type = {}
        for i, data in enumerate(batch_data):
            request = InferenceRequest(**data)
            if request.model_type not in requests_by_type:
                requests_by_type[request.model_type] = []
            requests_by_type[request.model_type].append((i, request))
        
        # Process each model type
        for model_type, requests in requests_by_type.items():
            # Process requests in parallel
            tasks = []
            for idx, request in requests:
                task = self.session_pool.run_inference(request)
                tasks.append((idx, task))
            
            # Wait for all to complete
            for idx, task in tasks:
                result = await task
                results.append({
                    "request_id": result.request_id,
                    "model_type": result.model_type,
                    "outputs": result.outputs,
                    "processing_time": result.processing_time,
                    "session_id": result.session_id
                })
        
        return results
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        stats = {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "enable_batching": self.enable_batching,
            "batch_strategy": self.batch_strategy.value if self.batcher else None,
            "pool_stats": self.session_pool.get_pool_stats()
        }
        
        if self.batcher:
            stats["batcher_metrics"] = self.batcher.get_metrics()
            stats["batcher_queue_status"] = self.batcher.get_queue_status()
        
        return stats

# Test function
async def test_runtime_engine():
    """Test the runtime engine"""
    print("Testing Runtime Engine...")
    
    # Mock model paths (in real usage, these would be actual ONNX files)
    model_paths = {
        "text_encoder": "/tmp/text_encoder.onnx",
        "image_encoder": "/tmp/image_encoder.onnx",
        "fusion_head": "/tmp/fusion_head.onnx"
    }
    
    # Create engine
    engine = RuntimeEngine(
        model_paths=model_paths,
        pool_size=2,
        batch_strategy=BatchStrategy.ADAPTIVE,
        enable_batching=True
    )
    
    try:
        # Initialize engine
        await engine.initialize()
        
        # Create dummy inputs
        input_ids = np.random.randint(0, 30000, (1, 512), dtype=np.int64)
        attention_mask = np.ones((1, 512), dtype=np.int64)
        image = np.random.random((1, 3, 224, 224)).astype(np.float32)
        features = np.random.random((1, 1280)).astype(np.float32)
        
        # Run inference
        print("Running text encoder inference...")
        text_result = await engine.run_text_encoder(input_ids, attention_mask)
        print(f"Text result shape: {text_result.shape}")
        
        print("Running image encoder inference...")
        image_result = await engine.run_image_encoder(image)
        print(f"Image result shape: {image_result.shape}")
        
        print("Running fusion head inference...")
        fusion_result = await engine.run_fusion_head(features)
        print(f"Fusion result shape: {fusion_result.shape}")
        
        # Get stats
        stats = engine.get_engine_stats()
        print(f"Engine stats: {stats}")
        
    finally:
        # Shutdown engine
        await engine.shutdown()
    
    print("Runtime Engine test completed!")

if __name__ == "__main__":
    asyncio.run(test_runtime_engine())
