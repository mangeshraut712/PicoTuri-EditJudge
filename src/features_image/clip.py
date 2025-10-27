"""
CLIP Image Embeddings Pipeline
Advanced image embeddings using OpenAI CLIP for PicoTuri-EditJudge
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open_clip
from typing import List, Optional, Union, Tuple
import logging
from pathlib import Path
import time
import cv2

logger = logging.getLogger(__name__)

class CLIPImageEmbedder:
    """
    CLIP-based image embedding system with similarity computation
    Supports ViT-B/32 and ViT-L/14 models
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        batch_size: int = 16,
        image_size: int = 224,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize CLIP image embedder

        Args:
            model_name: CLIP model architecture (ViT-B-32 or ViT-L-14)
            pretrained: Pretrained weights (openai, laion2b_s32b_b79k)
            device: Device to run inference on
            batch_size: Default batch size for inference
            image_size: Input image size
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        self.model_config = {
            "ViT-B-32": {
                "embed_dim": 512,
                "image_resolution": 224,
                "vision_layers": 12,
                "vision_width": 768,
                "vision_patch_size": 32
            },
            "ViT-L-14": {
                "embed_dim": 768,
                "image_resolution": 224,
                "vision_layers": 24,
                "vision_width": 1024,
                "vision_patch_size": 14
            }
        }

        # Load model and preprocessing
        self._load_model(cache_dir)

        # Set hidden size based on model
        self.hidden_size = self.model_config[self.model_name]["embed_dim"]

        # Performance tracking
        self.inference_times = []

    def _load_model(self, cache_dir: Optional[str] = None):
        """Load CLIP model and preprocessing"""
        try:
            logger.info(f"Loading CLIP model: {self.model_name}/{self.pretrained}")

            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
                cache_dir=cache_dir
            )

            self.model.eval()

            logger.info(f"CLIP model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Load images from paths with error handling"""
        images = []

        for path in image_paths:
            try:
                # Load image
                image = Image.open(path)

                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                images.append(image)

            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                # Create a dummy black image as fallback
                images.append(Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0)))

        return images

    def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess images for CLIP"""
        preprocessed = []

        for image in images:
            # Apply CLIP preprocessing
            processed = self.preprocess(image)
            preprocessed.append(processed)

        # Stack into batch
        return torch.stack(preprocessed).to(self.device)

    def embed_image(self, image_paths: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for a list of images

        Args:
            image_paths: List of image file paths
            batch_size: Override default batch size

        Returns:
            numpy array of shape (len(image_paths), embed_dim)
        """
        if not image_paths:
            return np.array([]).reshape(0, self.get_embedding_dim())

        batch_size = batch_size or self.batch_size

        # Load images
        images = self._load_images(image_paths)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            start_time = time.time()

            # Preprocess images
            image_tensor = self._preprocess_images(batch_images)

            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)

                # Normalize embeddings
                image_features = F.normalize(image_features, p=2, dim=-1)

            all_embeddings.append(image_features.cpu().numpy())

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time / len(batch_images))

        # Concatenate all batches
        final_embeddings = np.concatenate(all_embeddings, axis=0)

        logger.info(f"Generated embeddings for {len(image_paths)} images, shape: {final_embeddings.shape}")
        return final_embeddings

    def embed_single(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image"""
        return self.embed_image([image_path])[0]

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity matrix of shape (len(embeddings1), len(embeddings2))
        """
        # Normalize embeddings
        embeddings1_norm = F.normalize(torch.from_numpy(embeddings1), p=2, dim=-1)
        embeddings2_norm = F.normalize(torch.from_numpy(embeddings2), p=2, dim=-1)

        # Compute cosine similarity
        similarity = torch.mm(embeddings1_norm, embeddings2_norm.T)

        return similarity.numpy()

    def compute_image_text_similarity(self, image_paths: List[str], texts: List[str]) -> np.ndarray:
        """
        Compute image-text similarity using CLIP

        Args:
            image_paths: List of image file paths
            texts: List of text strings

        Returns:
            Similarity matrix of shape (len(image_paths), len(texts))
        """
        # Get image embeddings
        image_embeddings = self.embed_image(image_paths)

        # Get text embeddings
        text_embeddings = self._encode_text(texts)

        # Compute similarity
        return self.compute_similarity(image_embeddings, text_embeddings)

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts using CLIP text encoder"""
        try:
            # Tokenize texts
            text_tokens = open_clip.tokenize(texts).to(self.device)

            # Encode texts
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=-1)

            return text_features.cpu().numpy()

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model_config.get(self.model_name, {}).get("embed_dim", 512)

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
        """Export vision encoder to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, self.image_size, self.image_size).to(self.device)

            # Export to ONNX
            torch.onnx.export(
                self.model.visual,
                dummy_input,
                output_path,
                input_names=['image'],
                output_names=['image_features'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'image_features': {0: 'batch_size'}
                },
                opset_version=14
            )

            logger.info(f"CLIP vision encoder exported to ONNX: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            raise

# Convenience function for quick usage
def embed_image(image_paths: List[str], model_name: str = "ViT-B-32") -> np.ndarray:
    """Quick image embedding function"""
    embedder = CLIPImageEmbedder(model_name=model_name)
    return embedder.embed_image(image_paths)

# Test function
def test_clip_embeddings():
    """Test CLIP embeddings functionality"""
    print("Testing CLIP embeddings...")

    # Create dummy images for testing
    test_images = []
    for i in range(3):
        # Create random colored images
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = f"/tmp/test_image_{i}.jpg"
        img.save(img_path)
        test_images.append(img_path)

    try:
        # Test with ViT-B-32
        clip_embedder = CLIPImageEmbedder("ViT-B-32")
        embeddings = clip_embedder.embed_image(test_images)

        print(f"CLIP embeddings shape: {embeddings.shape}")
        print(f"CLIP embedding dim: {clip_embedder.get_embedding_dim()}")
        print(f"CLIP performance: {clip_embedder.get_performance_stats()}")

        # Test similarity computation
        similarity_matrix = clip_embedder.compute_similarity(embeddings, embeddings)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")

        # Test image-text similarity
        texts = ["A beautiful landscape", "A portrait photo", "An abstract image"]
        img_text_sim = clip_embedder.compute_image_text_similarity(test_images, texts)
        print(f"Image-text similarity shape: {img_text_sim.shape}")

        # Test with ViT-L-14
        try:
            clip_embedder_l14 = CLIPImageEmbedder("ViT-L-14")
            embeddings_l14 = clip_embedder_l14.embed_image(test_images)

            print(f"CLIP-L14 embeddings shape: {embeddings_l14.shape}")
            print(f"CLIP-L14 embedding dim: {clip_embedder_l14.get_embedding_dim()}")

        except Exception as e:
            print(f"ViT-L-14 test failed: {e}")

    except Exception as e:
        print(f"CLIP test failed: {e}")

    finally:
        # Clean up test images
        for img_path in test_images:
            try:
                Path(img_path).unlink()
            except Exception:
                pass

    print("CLIP embeddings test completed!")

if __name__ == "__main__":
    test_clip_embeddings()
