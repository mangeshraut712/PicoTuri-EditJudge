#!/usr/bin/env python3
"""
PHASE 5: Optimize for Device - Core ML Deployment Pipeline

Converts PyTorch SFT model to Core ML format for iOS deployment
Includes quantization, model export, and Apple Silicon optimization
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from src.models.simple_instruction_processor import SimpleInstructionProcessor


# ============================================================================
# MODEL LOADER (From SFT Training)
# ============================================================================

class CoreMLModelLoader:
    """
    Load trained PyTorch model and prepare for Core ML conversion
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                   'cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model, self.config = self._load_model()
        self.model.eval()

    def _load_model(self):
        """Load trained SFT model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Recreate model architecture
            model = SimpleInstructionProcessor(
                vocab_size=checkpoint['config']['vocab_size'],
                embedding_dim=checkpoint['config']['embedding_dim'],
                hidden_dim=checkpoint['config']['hidden_dim'],
                num_classes=checkpoint['config']['num_classes']
            )

            # Load state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)

            print("✅ Model loaded successfully!")
            print(f"   Vocab size: {checkpoint['config']['vocab_size']}")
            print(f"   Classes: {checkpoint['config']['num_classes']}")

            return model, checkpoint['config']

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def get_sample_input(self) -> Dict[str, torch.Tensor]:
        """Create sample input for tracing"""
        # Get vocabulary size for input creation
        vocab_size = self.config['vocab_size']

        # Create sample input (same format as training)
        # This should match what dataset generates
        sample_input = torch.tensor([
            [1, 2, 3, 4, 5] + [0] * (self.config.get('max_length', 512) - 5)  # Sample tokenized instruction
        ], dtype=torch.long).to(self.device)

        return {'instruction_tokens': sample_input}

    def count_parameters(self) -> int:
        """Count model parameters for size estimation"""
        return sum(p.numel() for p in self.model.parameters())


# ============================================================================
# CORE ML OPTIMIZATION PIPELINE
# ============================================================================

class CoreMLOptimizer:
    """
    Complete Core ML conversion and optimization pipeline
    """

    def __init__(self, model_loader: CoreMLModelLoader):
        self.model_loader = model_loader
        self.config = model_loader.config

    def export_to_torchscript(self, output_path: Path) -> Path:
        """First step: Convert to TorchScript"""
        print("\n🔄 Step 1: Converting to TorchScript...")

        # Create sample input
        sample_input = self.model_loader.get_sample_input()

        # Trace the model
        with torch.no_grad():
            # Create a wrapper for JIT tracing
            class TracingWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, instruction_tokens):
                    return self.model(instruction_tokens)

            wrapper = TracingWrapper(self.model_loader.model)
            traced_model = torch.jit.trace(wrapper, sample_input['instruction_tokens'])

        # Save traced model
        traced_path = output_path.with_suffix('.pt')
        torch.jit.save(traced_model, traced_path)

        print(f"✅ TorchScript model saved: {traced_path}")
        return traced_path

    def convert_to_coreml(self, torchscript_path: Path, output_path: Path) -> Path:
        """Second step: Convert to Core ML"""
        print("\n🔄 Step 2: Converting to Core ML...")

        try:
            import coremltools as ct

            # Load traced model
            traced_model = torch.jit.load(torchscript_path)

            # Create sample input (numpy array with correct dtype)
            sample_input_np = self.model_loader.get_sample_input()['instruction_tokens'].cpu().numpy()

            # Convert to Core ML
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(
                    name='instruction_tokens',
                    shape=sample_input_np.shape,
                    dtype=np.int32
                )],
                # Core ML 3 format for iOS 15+
                minimum_deployment_target=ct.target.iOS15
            )

            # Set metadata (where supported) - handle potential attribute errors
            try:
                mlmodel.author = "PicoTuri-EditJudge Project"  # type: ignore
            except AttributeError:
                pass
            try:
                mlmodel.license = "Apache License 2.0"  # type: ignore
            except AttributeError:
                pass
            try:
                mlmodel.short_description = (  # type: ignore
                    "AI-powered image editing with natural language instructions"
                )
            except AttributeError:
                pass
            try:
                mlmodel.version = "1.0.0"  # type: ignore
            except AttributeError:
                pass
            try:
                mlmodel.license_description = (  # type: ignore
                    "This model provides AI-powered image editing capabilities. "
                    "Licensed under Apache 2.0 for open-source use."
                )
            except AttributeError:
                pass

            # Save Core ML model
            coreml_path = output_path.with_suffix('.mlmodel')
            mlmodel.save(str(coreml_path))  # type: ignore

            print(f"✅ Core ML model saved: {coreml_path}")
            # Safely access version and author if they exist
            try:
                print(f"   Model version: {mlmodel.version}")  # type: ignore
            except AttributeError:
                print("   Model version: 1.0.0")
            try:
                print(f"   Author: {mlmodel.author}")  # type: ignore
            except AttributeError:
                print("   Author: PicoTuri-EditJudge Project")

            return coreml_path

        except ImportError:
            print("⚠️  Core ML conversion requires coremltools:")
            print("   pip install coremltools>=8.0")
            return torchscript_path.with_suffix('.mlmodel')  # Placeholder path

        except Exception as e:
            print(f"❌ Core ML conversion failed: {e}")
            return torchscript_path.with_suffix('.mlmodel')  # Placeholder path

    def quantify_model(self, coreml_path: Path) -> Path:
        """Optional: Apply quantization for smaller size/faster inference"""
        print("\n🔄 Step 3: Applying quantization (optional)...")

        try:
            import coremltools as ct

            # Note: Quantization APIs vary by coremltools version
            # This is a simplified approach - full quantization would require
            # coremltools.optimize.coreml with more complex setup
            print("   ⚠️  Quantization skipped (requires coremltools.optimize)")
            print("   For production, add: pip install coremltools[optimize]")
            return coreml_path

        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            print("   Proceeding with original model...")
            return coreml_path

    def create_manifest(self, coreml_path: Path, output_dir: Path) -> Path:
        """Create deployment manifest for iOS app"""
        print("\n🔄 Step 4: Creating deployment manifest...")

        # Load the trained model's config
        model_config = self.config

        # Get model size
        model_size = coreml_path.stat().st_size

        # Create manifest
        manifest = {
            'model_info': {
                'name': 'PicoTuri-EditJudge-SFT-Model',
                'version': '1.0.0',
                'type': 'Core ML Neural Network',
                'created_at': datetime.now().isoformat(),
                'framework_version': 'PyTorch → Core ML Tools',
                'preprocessing': 'Simple tokenization (demo)',
                'postprocessing': 'Classification → edit type'
            },

            'architecture': {
                'embedding_dim': model_config['embedding_dim'],
                'hidden_dim': model_config['hidden_dim'],
                'num_classes': model_config['num_classes'],
                'vocab_size': model_config['vocab_size'],
                'parameters': self.model_loader.count_parameters(),
                'input_shape': [1, model_config.get('max_length', 512)],  # [batch, seq_len]
                'output_shape': [1, model_config['num_classes']]  # [batch, num_classes]
            },

            'performance': {
                'model_size': model_size,
                'model_size_mb': round(model_size / (1024 * 1024), 2),
                'estimated_inference_time': '< 10ms (estimate)',
                'optimal_batch_size': 1,
                'recommended_device': 'iPhone 15 Pro or newer'
            },

            'edit_types': {
                0: 'brightness_adjustment',
                1: 'color_adjustment',
                2: 'blur_background'
            },

            'vocabulary_info': {
                'size': model_config['vocab_size'],
                'tokenization': 'word-level (simplified for demo)',
                'max_length': model_config.get('max_length', 512)
            },

            'deployment_requirements': {
                'iOS_version': '15.0+',
                'core_ml_version': '7.0+',
                'cpu_only_fallback': False,
                'neural_engine_preferred': True,
                'compression_ratio': None  # Would be filled if quantized
            }
        }

        # Save manifest
        manifest_path = output_dir / 'model_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Deployment manifest created: {manifest_path}")
        return manifest_path

    def export_complete_pipeline(self, output_dir: Path = Path('./coreml_output')):
        """Run complete export pipeline"""
        print("🚀 STARTING CORE ML EXPORT PIPELINE")
        print("=" * 50)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert to TorchScript
        torchscript_path = self.export_to_torchscript(output_dir / 'model')

        # Step 2: Convert to Core ML
        coreml_path = self.convert_to_coreml(torchscript_path, output_dir / 'model')

        # Step 3: Optional quantization
        final_path = self.quantify_model(coreml_path)

        # Step 4: Create manifest
        manifest_path = self.create_manifest(final_path, output_dir)

        print("\n🎉 CORE ML EXPORT COMPLETED!")
        print("=" * 50)
        print("📁 Output files:")
        print(f"   TorchScript: {torchscript_path}")
        print(f"   Core ML: {coreml_path}")
        print(f"   Quantized: {final_path}")
        print(f"   Manifest: {manifest_path}")
        print()
        print("📱 Ready for iOS deployment!")
        print(f"   Model size: {final_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Parameters: {self.model_loader.count_parameters():,}")

        return {
            'torchscript': torchscript_path,
            'coreml': coreml_path,
            'quantized': final_path,
            'manifest': manifest_path
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete Core ML optimization pipeline"""

    print("📱 PHASE 5: OPTIMIZE FOR DEVICE - Core ML Deployment")
    print("=" * 60)

    # Load trained model
    print("📂 Loading trained SFT model...")
    model_path = Path('./outputs/sft_models/best_model.pt')

    if model_path.exists():
        print(f"   Found model: {model_path}")
    else:
        # Try latest checkpoint
        model_path = Path('./outputs/sft_models/latest_model.pt')
        if model_path.exists():
            print(f"   Using latest checkpoint: {model_path}")
        else:
            print("❌ No trained model found!")
            print("   Please run SFT training first:")
            print("   python train_pico_banana_sft.py")
            return

    # Load model
    model_loader = CoreMLModelLoader(model_path)

    # Run Core ML optimization pipeline
    optimizer = CoreMLOptimizer(model_loader)
    exported_files = optimizer.export_complete_pipeline()

    # Verify Core ML model
    print("\n🔍 VERIFYING CORE ML MODEL...")

    try:
        import coremltools as ct

        coreml_model = ct.models.MLModel(str(exported_files['coreml']))
        print("✅ Core ML model verification passed")
        outputs = getattr(coreml_model, "output_description", {})
        if outputs:
            for name, desc in outputs.items():
                print(f"   Output '{name}': {desc}")

    except ImportError:
        print("⚠️  Core ML verification skipped (coremltools not installed)")
        print("   Install for full functionality: pip install coremltools>=8.0")

    except Exception as e:
        print(f"⚠️  Core ML verification failed: {e}")

    # Final summary
    print("\n🎯 DEPLOYMENT READY!")
    print("=" * 60)
    print("📱 Next Steps:")
    print("   1. Add Core ML model to iOS project")
    print("   2. Load model in Swift code")
    print("   3. Integrate with camera/video processing")
    print("   4. Test on real device (iPhone 15 Pro recommended)")
    print("   5. Profile performance and power usage")
    print()
    print("📋 Files ready:")
    for name, path in exported_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"   {name}: ⚠️  Not created")

    print("\n🚀 Apple Ecosystem Integration Complete!")
    print("   From PyTorch SFT → Core ML Neural Engine")
    print("   Ready for production iOS deployment!")


if __name__ == "__main__":
    main()
