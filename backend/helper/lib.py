from typing import Optional
from tangoflux import TangoFluxInference
import threading
from parler_tts import ParlerTTSForConditionalGeneration
import os
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Thread-local storage for worker IDs
_thread_local = threading.local()

class TangoFluxModel:
    """Manages TangoFlux model instances with support for parallel execution.
    
    Note: TangoFlux models are NOT thread-safe. For parallel execution, we maintain
    a pool of model instances (one per worker) to avoid race conditions.
    For sequential execution, we use a single instance with a lock.
    """

    _instance = None
    _lock = threading.Lock()
    _generate_lock = threading.Lock()  # Lock for serializing generate() calls in sequential mode
    _model_pool = []  # Pool of model instances for parallel execution
    _pool_lock = threading.Lock()
    _pool_size = 0

    @classmethod
    def get_instance(cls):
        """Get the primary singleton instance (for sequential mode)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TangoFluxInference(name="declare-lab/TangoFlux")
        return cls._instance
    
    @classmethod
    def _get_model_from_pool(cls, worker_id: int):
        """Get a model instance from the pool for parallel execution."""
        with cls._pool_lock:
            # Ensure pool is large enough
            while len(cls._model_pool) <= worker_id:
                cls._model_pool.append(TangoFluxInference(name="declare-lab/TangoFlux"))
            return cls._model_pool[worker_id]
    
    @classmethod
    def initialize_pool(cls, pool_size: int):
        """Pre-initialize model pool for parallel execution.
        
        This method is idempotent - it only creates new models if the pool
        is smaller than the requested size. Safe to call multiple times.
        """
        with cls._pool_lock:
            current_size = len(cls._model_pool)
            if current_size >= pool_size:
                # Pool already has enough models
                return
            
            cls._pool_size = max(cls._pool_size, pool_size)
            # Pre-create models up to pool_size
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Initializing TangoFlux model pool: {current_size} -> {pool_size} models")
            while len(cls._model_pool) < pool_size:
                cls._model_pool.append(TangoFluxInference(name="declare-lab/TangoFlux"))
            logger.info(f"TangoFlux model pool initialized with {len(cls._model_pool)} models")
    
    @classmethod
    def _get_current_worker_id(cls):
        """Get worker ID from thread-local storage if available."""
        return getattr(_thread_local, 'worker_id', None)
    
    @classmethod
    def generate(cls, prompt: str, steps: int, duration: int, worker_id: Optional[int] = None):
        """Generate audio with thread-safe handling.
        
        Args:
            prompt: Text prompt for generation
            steps: Number of diffusion steps
            duration: Duration in seconds
            worker_id: Optional worker ID for parallel execution (uses pool)
                      If None, checks thread-local storage
        
        If worker_id is provided or found in thread-local, uses model from pool (parallel mode).
        Otherwise, uses singleton with lock (sequential mode).
        """
        # Check thread-local if worker_id not provided
        if worker_id is None:
            worker_id = cls._get_current_worker_id()
        
        if worker_id is not None:
            # Parallel mode: use model from pool
            model = cls._get_model_from_pool(worker_id)
            return model.generate(prompt, steps=steps, duration=duration)
        else:
            # Sequential mode: use singleton with lock
            model = cls.get_instance()
            with cls._generate_lock:
                return model.generate(prompt, steps=steps, duration=duration)
    
class ParlerTTSModel:
    """Singleton to manage ParlerTTS model and tokenizers."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    HF_TOKEN = (
                        os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                        or os.getenv("HF_TOKEN")
                        or os.getenv("HUGGING_FACE_HUB_TOKEN")
                    )
                    model = ParlerTTSForConditionalGeneration.from_pretrained(
                        "ai4bharat/indic-parler-tts", token=HF_TOKEN
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        "ai4bharat/indic-parler-tts", token=HF_TOKEN
                    )
                    description_tokenizer = AutoTokenizer.from_pretrained(
                        model.config.text_encoder._name_or_path, token=HF_TOKEN
                    )
                    cls._instance = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "description_tokenizer": description_tokenizer,
                    }
        return cls._instance