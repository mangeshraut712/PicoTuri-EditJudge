"""
BERT Text Embeddings Pipeline
Advanced text embeddings using BERT for PicoTuri-EditJudge
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Union
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class BERTTextEmbedder:
    """
    BERT-based text embedding system with sentence-level pooling
    Supports bert-base-uncased and e5-small-v2 models
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize BERT text embedder

        Args:
            model_name: Model identifier (bert-base-uncased or e5-small-v2)
            device: Device to run inference on
            max_length: Maximum sequence length
            batch_size: Default batch size for inference
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        self.model_config = {
            "bert-base-uncased": {
                "hidden_size": 768,
                "pooling": "cls",
                "normalize": True
            },
            "e5-small-v2": {
                "hidden_size": 384,
                "pooling": "mean",
                "normalize": True,
                "prefix": "passage: "
            }
        }

        # Load tokenizer and model
        self._load_model(cache_dir)

        # Performance tracking
        self.inference_times = []

    def _load_model(self, cache_dir: Optional[str] = None):
        """Load tokenizer and model"""
        try:
            logger.info(f"Loading BERT model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                use_fast=True
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocess texts according to model requirements"""
        config = self.model_config.get(self.model_name, {})

        processed_texts = []
        for text in texts:
            # Add prefix for e5 models
            if "prefix" in config:
                text = config["prefix"] + text

            # Basic cleaning
            text = text.strip()
            processed_texts.append(text)

        return processed_texts

    def _pool_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings to sentence level"""
        config = self.model_config.get(self.model_name, {})
        pooling_strategy = config.get("pooling", "cls")

        if pooling_strategy == "cls":
            # CLS token pooling
            return embeddings[:, 0]

        elif pooling_strategy == "mean":
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            return masked_embeddings.sum(1) / mask.sum(1)

        elif pooling_strategy == "max":
            # Max pooling with attention mask
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings.masked_fill(mask == 0, -1e9)
            return torch.max(masked_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    def embed_text(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            batch_size: Override default batch size

        Returns:
            numpy array of shape (len(texts), hidden_size)
        """
        if not texts:
            return np.array([]).reshape(0, self.get_embedding_dim())

        batch_size = batch_size or self.batch_size
        texts = self._preprocess_text(texts)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            start_time = time.time()

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._pool_embeddings(outputs.last_hidden_state, inputs['attention_mask'])

                # Normalize if required
                config = self.model_config.get(self.model_name, {})
                if config.get("normalize", False):
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time / len(batch_texts))

        # Concatenate all batches
        final_embeddings = np.concatenate(all_embeddings, axis=0)

        logger.info(f"Generated embeddings for {len(texts)} texts, shape: {final_embeddings.shape}")
        return final_embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embed_text([text])[0]

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text (alias for embed_single for compatibility)"""
        return self.embed_single(text)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model_config.get(self.model_name, {}).get("hidden_size", 768)

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        return {
            "avg_time_per_sample": np.mean(self.inference_times) * 1000,  # ms
            "p95_time_per_sample": np.percentile(self.inference_times, 95) * 1000,
            "total_samples": len(self.inference_times),
            "device": self.device
        }

    def save_onnx(self, output_path: str, batch_size: int = 1):
        """Export model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = self.tokenizer(
                ["dummy text"] * batch_size,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}

            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                },
                opset_version=14
            )

            logger.info(f"Model exported to ONNX: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            raise

# Convenience function for quick usage
def embed_text(texts: List[str], model_name: str = "bert-base-uncased") -> np.ndarray:
    """Quick text embedding function"""
    embedder = BERTTextEmbedder(model_name=model_name)
    return embedder.embed_text(texts)

# Test function
def test_bert_embeddings():
    """Test BERT embeddings functionality"""
    print("Testing BERT embeddings...")

    # Test texts
    texts = [
        "This is a test sentence for BERT embeddings.",
        "Another example to test the embedding functionality.",
        "Make the image more vibrant and colorful.",
        "Remove the background from this photo."
    ]

    # Test with BERT
    bert_embedder = BERTTextEmbedder("bert-base-uncased")
    bert_embeddings = bert_embedder.embed_text(texts)

    print(f"BERT embeddings shape: {bert_embeddings.shape}")
    print(f"BERT embedding dim: {bert_embedder.get_embedding_dim()}")
    print(f"BERT performance: {bert_embedder.get_performance_stats()}")

    # Test with e5-small-v2
    try:
        e5_embedder = BERTTextEmbedder("intfloat/e5-small-v2")
        e5_embeddings = e5_embedder.embed_text(texts)

        print(f"E5 embeddings shape: {e5_embeddings.shape}")
        print(f"E5 embedding dim: {e5_embedder.get_embedding_dim()}")
        print(f"E5 performance: {e5_embedder.get_performance_stats()}")

    except Exception as e:
        print(f"E5 test failed: {e}")

    print("BERT embeddings test completed!")

if __name__ == "__main__":
    test_bert_embeddings()
