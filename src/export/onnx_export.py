"""
ONNX Export Script
Export BERT, CLIP, and fusion models to ONNX format for cross-platform deployment
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging
import yaml
from typing import Dict, List, Optional, Tuple
import time

# Import our modules
from src.features_text.bert import BERTTextEmbedder
from src.features_image.clip import CLIPImageEmbedder
from src.fuse.stack import FeatureFusionStack, FusionMLPHead

logger = logging.getLogger(__name__)

class ONNXExporter:
    """
    Export models to ONNX format with optimization and validation
    """
    
    def __init__(self, config_path: str = "configs/models/bert_clip.yaml"):
        """
        Initialize ONNX exporter
        
        Args:
            config_path: Path to model configuration
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def export_text_encoder(self) -> str:
        """
        Export BERT text encoder to ONNX
        
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting BERT text encoder to ONNX...")
        
        # Initialize BERT embedder
        text_config = self.config['text']
        embedder = BERTTextEmbedder(
            model_name=text_config['model_name'],
            device="cpu",  # Export on CPU for compatibility
            max_length=text_config['max_length'],
            batch_size=text_config['batch_size']
        )
        
        # Export configuration
        onnx_config = self.config['onnx']['text_encoder']
        output_path = self.output_dir / onnx_config['output_path'].split('/')[-1]
        
        # Create dummy input
        batch_size = onnx_config['batch_size']
        dummy_texts = ["dummy text"] * batch_size
        
        # Tokenize
        inputs = embedder.tokenizer(
            dummy_texts,
            padding=True,
            truncation=True,
            max_length=text_config['max_length'],
            return_tensors="pt"
        )
        
        # Export to ONNX
        try:
            torch.onnx.export(
                embedder.model,
                (inputs['input_ids'], inputs['attention_mask']),
                output_path,
                input_names=onnx_config['input_names'],
                output_names=onnx_config['output_names'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                },
                opset_version=self.config['onnx']['opset_version'],
                do_constant_folding=True
            )
            
            logger.info(f"Text encoder exported to {output_path}")
            
            # Validate ONNX model
            self._validate_onnx_model(output_path)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export text encoder: {e}")
            raise
    
    def export_image_encoder(self) -> str:
        """
        Export CLIP image encoder to ONNX
        
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting CLIP image encoder to ONNX...")
        
        # Initialize CLIP embedder
        image_config = self.config['image']
        embedder = CLIPImageEmbedder(
            model_name=image_config['model_name'],
            pretrained=image_config['pretrained'],
            device="cpu",  # Export on CPU for compatibility
            batch_size=image_config['batch_size'],
            image_size=image_config['image_size']
        )
        
        # Export configuration
        onnx_config = self.config['onnx']['image_encoder']
        output_path = self.output_dir / onnx_config['output_path'].split('/')[-1]
        
        # Create dummy input
        batch_size = onnx_config['batch_size']
        dummy_input = torch.randn(batch_size, 3, image_config['image_size'], image_config['image_size'])
        
        # Export to ONNX
        try:
            torch.onnx.export(
                embedder.model.visual,
                dummy_input,
                output_path,
                input_names=onnx_config['input_names'],
                output_names=onnx_config['output_names'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'image_features': {0: 'batch_size'}
                },
                opset_version=self.config['onnx']['opset_version'],
                do_constant_folding=True
            )
            
            logger.info(f"Image encoder exported to {output_path}")
            
            # Validate ONNX model
            self._validate_onnx_model(output_path)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export image encoder: {e}")
            raise
    
    def export_fusion_head(self) -> str:
        """
        Export fusion head to ONNX
        
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting fusion head to ONNX...")
        
        # Initialize fusion head
        fusion_config = self.config['fusion']
        feature_dims = fusion_config['feature_dims']
        total_input_dim = sum(feature_dims.values())
        
        mlp_config = fusion_config['mlp']
        model = FusionMLPHead(
            input_dim=total_input_dim,
            hidden_dims=mlp_config['hidden_dims'],
            output_dim=mlp_config['output_dim'],
            dropout_rate=mlp_config['dropout_rate'],
            activation=mlp_config['activation'],
            use_batch_norm=mlp_config['use_batch_norm'],
            use_residual=mlp_config['use_residual']
        )
        
        # Export configuration
        onnx_config = self.config['onnx']['fusion_head']
        output_path = self.output_dir / onnx_config['output_path'].split('/')[-1]
        
        # Create dummy input
        batch_size = onnx_config['batch_size']
        dummy_input = torch.randn(batch_size, total_input_dim)
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=onnx_config['input_names'],
                output_names=onnx_config['output_names'],
                dynamic_axes={
                    'features': {0: 'batch_size'},
                    'scores': {0: 'batch_size'}
                },
                opset_version=self.config['onnx']['opset_version'],
                do_constant_folding=True
            )
            
            logger.info(f"Fusion head exported to {output_path}")
            
            # Validate ONNX model
            self._validate_onnx_model(output_path)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export fusion head: {e}")
            raise
    
    def _validate_onnx_model(self, model_path: str):
        """
        Validate ONNX model with ONNX Runtime
        
        Args:
            model_path: Path to ONNX model
        """
        try:
            # Load model
            ort_session = ort.InferenceSession(model_path)
            
            # Get input info
            input_info = ort_session.get_inputs()
            output_info = ort_session.get_outputs()
            
            logger.info(f"Model inputs: {[inp.name for inp in input_info]}")
            logger.info(f"Model outputs: {[out.name for out in output_info]}")
            
            # Test inference with dummy data
            dummy_inputs = {}
            for inp in input_info:
                if len(inp.shape) == 2:  # Text input
                    dummy_inputs[inp.name] = np.random.randint(0, 1000, inp.shape, dtype=np.int64)
                elif len(inp.shape) == 4:  # Image input
                    dummy_inputs[inp.name] = np.random.randn(*inp.shape).astype(np.float32)
                else:  # Features input
                    dummy_inputs[inp.name] = np.random.randn(*inp.shape).astype(np.float32)
            
            # Run inference
            outputs = ort_session.run(None, dummy_inputs)
            
            logger.info(f"Model validation successful. Output shapes: {[out.shape for out in outputs]}")
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise
    
    def optimize_onnx_model(self, model_path: str) -> str:
        """
        Optimize ONNX model for better performance
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Path to optimized model
        """
        try:
            # Load ONNX model
            onnx_model = onnx.load(model_path)
            
            # Optimize model (basic optimization)
            # For more advanced optimization, consider using onnx-simplifier
            optimized_model = onnx.shape_inference.infer_shapes(onnx_model)
            
            # Save optimized model
            optimized_path = model_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            logger.info(f"Model optimized and saved to {optimized_path}")
            return optimized_path
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model_path
    
    def export_all_models(self) -> Dict[str, str]:
        """
        Export all models to ONNX format
        
        Returns:
            Dictionary of model names to file paths
        """
        logger.info("Starting ONNX export for all models...")
        
        exported_models = {}
        
        try:
            # Export text encoder
            text_encoder_path = self.export_text_encoder()
            exported_models['text_encoder'] = text_encoder_path
            
            # Export image encoder
            image_encoder_path = self.export_image_encoder()
            exported_models['image_encoder'] = image_encoder_path
            
            # Export fusion head
            fusion_head_path = self.export_fusion_head()
            exported_models['fusion_head'] = fusion_head_path
            
            # Optimize models
            for name, path in exported_models.items():
                optimized_path = self.optimize_onnx_model(path)
                exported_models[f"{name}_optimized"] = optimized_path
            
            logger.info("All models exported successfully!")
            
            # Save export metadata
            self._save_export_metadata(exported_models)
            
            return exported_models
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def _save_export_metadata(self, exported_models: Dict[str, str]):
        """
        Save export metadata
        
        Args:
            exported_models: Dictionary of exported models
        """
        metadata = {
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'models': {}
        }
        
        for name, path in exported_models.items():
            if Path(path).exists():
                metadata['models'][name] = {
                    'path': path,
                    'size_mb': Path(path).stat().st_size / (1024 * 1024),
                    'optimized': 'optimized' in name
                }
        
        # Save metadata
        metadata_path = self.output_dir / "export_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Export metadata saved to {metadata_path}")

# Convenience function
def export_models_to_onnx(config_path: str = "configs/models/bert_clip.yaml") -> Dict[str, str]:
    """
    Export all models to ONNX format
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of exported model paths
    """
    exporter = ONNXExporter(config_path)
    return exporter.export_all_models()

# Test function
def test_onnx_export():
    """Test ONNX export functionality"""
    print("Testing ONNX export...")
    
    try:
        # Export models
        exported_models = export_models_to_onnx()
        
        print("Exported models:")
        for name, path in exported_models.items():
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"  {name}: {path} ({size_mb:.2f} MB)")
        
        # Test ONNX Runtime inference
        print("\nTesting ONNX Runtime inference...")
        
        # Load text encoder
        text_encoder_path = exported_models['text_encoder']
        text_session = ort.InferenceSession(text_encoder_path)
        
        # Create dummy input
        dummy_input_ids = np.random.randint(0, 1000, (1, 512), dtype=np.int64)
        dummy_attention_mask = np.ones((1, 512), dtype=np.int64)
        
        # Run inference
        text_outputs = text_session.run(None, {
            'input_ids': dummy_input_ids,
            'attention_mask': dummy_attention_mask
        })
        
        print(f"Text encoder output shape: {text_outputs[0].shape}")
        
        # Load image encoder
        image_encoder_path = exported_models['image_encoder']
        image_session = ort.InferenceSession(image_encoder_path)
        
        # Create dummy image input
        dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        image_outputs = image_session.run(None, {'image': dummy_image})
        
        print(f"Image encoder output shape: {image_outputs[0].shape}")
        
        # Load fusion head
        fusion_head_path = exported_models['fusion_head']
        fusion_session = ort.InferenceSession(fusion_head_path)
        
        # Create dummy features (concatenated text + image)
        total_features = 768 + 512  # BERT + CLIP
        dummy_features = np.random.randn(1, total_features).astype(np.float32)
        
        # Run inference
        fusion_outputs = fusion_session.run(None, {'features': dummy_features})
        
        print(f"Fusion head output shape: {fusion_outputs[0].shape}")
        
        print("ONNX export test completed!")
        
    except Exception as e:
        print(f"ONNX export test failed: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    test_onnx_export()
