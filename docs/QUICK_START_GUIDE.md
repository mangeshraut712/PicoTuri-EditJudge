# 🚀 Quick Start Guide - PicoTuri-EditJudge

## ⚡ Run All Tests (Recommended)

```bash
# Comprehensive verification of all algorithms
python test_all_algorithms.py
```

Expected output: **7/7 tests passed** ✅

---

## 🎯 Individual Algorithm Demos

### 1. Quality Scorer (4-Component System)
```bash
python src/algorithms/quality_scorer.py
```
**What it does:** Evaluates image edit quality using 4 weighted components:
- Instruction Compliance (40%)
- Editing Realism (25%)
- Preservation Balance (20%)
- Technical Quality (15%)

---

### 2. Diffusion Model (U-Net + Cross-Attention)
```bash
python src/algorithms/diffusion_model.py
```
**What it does:** Demonstrates instruction-guided image editing with:
- 10.9M parameter U-Net architecture
- Cross-attention for text conditioning
- DDPM/DDIM sampling support

---

### 3. DPO Training (Preference Optimization)
```bash
python -m src.algorithms.dpo_training
```
**What it does:** Shows preference-based model alignment:
- Bradley-Terry model loss
- KL divergence regularization
- Gradient-based optimization

---

### 4. Multi-Turn Editor (Conversational Editing)
```bash
python src/algorithms/multi_turn_editor.py
```
**What it does:** Demonstrates conversational image editing:
- Sequential instruction processing
- Context-aware editing
- Conflict detection
- Edit history tracking

---

### 5. Core ML Optimizer (Apple Silicon)
```bash
python -m src.algorithms.coreml_optimizer
```
**What it does:** Prepares models for iOS deployment:
- Core ML conversion
- Neural Engine optimization
- iOS integration code generation
- FP16 quantization

---

### 6. Dashboard (Visual Analytics)
```bash
python src/gui/dashboard.py
```
**What it does:** Generates performance visualizations:
- Metrics dashboard
- Dataset statistics
- Algorithm comparisons
- Prediction charts

---

## 📊 Quick Verification Commands

### Check All Modules Import Correctly
```bash
python -c "
from src.algorithms.quality_scorer import AdvancedQualityScorer
from src.algorithms.diffusion_model import AdvancedDiffusionModel
from src.algorithms.dpo_training import DPOTrainer
from src.algorithms.multi_turn_editor import MultiTurnEditor
from src.algorithms.coreml_optimizer import CoreMLOptimizer
from src.train.baseline import build_pipeline
print('✅ All modules imported successfully!')
"
```

### Run Individual Tests
```bash
# Test 1: Quality Scorer
echo "=== 1. QUALITY SCORER ===" && python src/algorithms/quality_scorer.py

# Test 2: Diffusion Model
echo "=== 2. DIFFUSION MODEL ===" && python src/algorithms/diffusion_model.py

# Test 3: DPO Training
echo "=== 3. DPO TRAINING ===" && python -m src.algorithms.dpo_training

# Test 4: Multi-Turn Editor
echo "=== 4. MULTI-TURN EDITOR ===" && python src/algorithms/multi_turn_editor.py

# Test 5: Core ML Optimizer
echo "=== 5. CORE ML OPTIMIZER ===" && python -m src.algorithms.coreml_optimizer
```

---

## 🔧 Training & Inference

### Train Baseline Model
```bash
python -m src.train.baseline \
    --pairs data/sample_dataset.csv \
    --model-path outputs/baseline_model.joblib \
    --test-size 0.3 \
    --seed 42
```

### Evaluate Existing Model
```bash
python -m src.train.baseline \
    --pairs data/sample_dataset.csv \
    --model-path outputs/baseline_model.joblib \
    --evaluate-only
```

---

## 📱 iOS Integration

### Generate iOS Code
```python
from src.algorithms.coreml_optimizer import iOSDeploymentManager

ios_manager = iOSDeploymentManager()
ios_files = ios_manager.generate_ios_integration_code(
    'PicoTuriEditJudge',
    'ios_output'
)
```

This generates:
- `ModelHandler.swift` - Core ML model interface
- `ImageProcessor.swift` - Image preprocessing
- `MainViewController.swift` - UI integration example

---

## 🐍 Python API Usage

### Quality Scorer
```python
from src.algorithms.quality_scorer import AdvancedQualityScorer
import torch

scorer = AdvancedQualityScorer()
original = torch.rand(1, 3, 512, 512)
edited = torch.rand(1, 3, 512, 512)
results = scorer(original, edited, ['brighten image'])

print(f"Overall Score: {results['overall_score']:.2f}")
```

### Diffusion Model
```python
from src.algorithms.diffusion_model import AdvancedDiffusionModel
import torch

model = AdvancedDiffusionModel(
    model_channels=64,
    channel_multipliers=[1, 2, 4]
)

x = torch.randn(1, 3, 64, 64)
t = torch.randint(0, 1000, (1,))
ctx = torch.randn(1, 16, 768)

output = model(x, t, ctx)
```

### DPO Training
```python
from src.algorithms.dpo_training import DPOTrainer
from src.algorithms.diffusion_model import AdvancedDiffusionModel
import torch

model = AdvancedDiffusionModel(model_channels=32, channel_multipliers=[1, 2])
ref_model = AdvancedDiffusionModel(model_channels=32, channel_multipliers=[1, 2])

trainer = DPOTrainer(model, ref_model, beta=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

accepted = torch.randn(4, 3, 32, 32)
rejected = torch.randn(4, 3, 32, 32)
instructions = ['brighten', 'darken', 'sharpen', 'blur']

metrics = trainer.train_step(accepted, rejected, instructions, optimizer)
```

### Multi-Turn Editor
```python
from src.algorithms.multi_turn_editor import MultiTurnEditor
import torch

editor = MultiTurnEditor()
initial_image = torch.rand(3, 256, 256)

instructions = [
    'brighten this photo',
    'increase contrast',
    'add blue filter'
]

results = editor.edit_conversationally(instructions, initial_image)
print(f"Success Rate: {results['session_summary']['overall_success_rate']:.1%}")
```

---

## 🔍 Troubleshooting

### Issue: Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements-dev.txt
```

### Issue: CUDA/MPS Not Available
```python
# All algorithms work on CPU - no GPU required for testing
device = 'cpu'  # Automatically handled
```

### Issue: Core ML Tools Warning
```
Torch version 2.9.0 has not been tested with coremltools
```
**Solution:** This is just a warning. Core ML functionality works correctly.

---

## 📈 Performance Benchmarks

### Expected Performance (Apple Silicon M1/M2/M3)

| Algorithm | Inference Time | Memory Usage |
|-----------|---------------|--------------|
| Quality Scorer | <100ms | ~200MB |
| Diffusion Model | <500ms | ~500MB |
| DPO Training | ~1s/step | ~1GB |
| Multi-Turn Editor | <50ms/edit | ~100MB |
| Baseline Model | <10ms | ~50MB |

---

## ✅ Verification Checklist

- [ ] Run `python test_all_algorithms.py` - All tests pass
- [ ] Run individual algorithm demos - All work correctly
- [ ] Check dashboard generation - PNG files created
- [ ] Verify iOS code generation - 3 Swift files created
- [ ] Test baseline training - Model saved successfully

---

## 🎯 Next Steps

1. **Explore the algorithms** - Run each demo to understand capabilities
2. **Train your own model** - Use your dataset with baseline training
3. **Deploy to iOS** - Use generated Swift code in Xcode
4. **Customize** - Modify algorithms for your specific use case

---

## 📚 Additional Resources

- **Full Documentation:** See `ALGORITHM_VERIFICATION_REPORT.md`
- **README:** See `README.md` for project overview
- **Contributing:** See `CONTRIBUTING.md` for guidelines
- **License:** Apache 2.0 (see `LICENSE`)

---

**Status:** ✅ All algorithms verified and working  
**Last Updated:** October 27, 2025  
**Ready for:** Production deployment on Apple Silicon
