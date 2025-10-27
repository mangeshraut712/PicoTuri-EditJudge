# Research Experiments Framework

This directory contains systematic research experiments for PicoTuri EditJudge, organized by research questions with reproducible protocols.

## Experiment Structure

```
experiments/
├── r1_embeddings/          # Embedding choice vs accuracy/latency
├── r2_fusion/             # Fusion architecture ablations  
├── r3_domain/             # Domain adaptation with LoRA
├── r4_preference/         # Preference learning for ranking
├── r5_robustness/         # Robustness & safety testing
├── r6_batching/           # Real-time batching & quantization
├── r7_parity/             # Cross-platform inference parity
├── configs/               # Shared experiment configurations
├── results/               # Aggregated results across experiments
└── plots/                 # Visualization outputs
```

## Running Experiments

### Single Experiment
```bash
# Run R1 embedding comparison
python scripts/run_experiment.py --config experiments/r1_embeddings/configs/bert_clip_b32.yaml

# Run with specific seeds
python scripts/run_experiment.py --config experiments/r1_embeddings/configs/e5_clip_l14.yaml --seeds 42,43,44
```

### Batch Experiments
```bash
# Run all R1 experiments
python scripts/run_experiment.py --suite r1_embeddings

# Run multiple research questions
python scripts/run_experiment.py --suite r1_embeddings,r2_fusion,r6_batching
```

### Benchmarking
```bash
# Latency benchmarking
python scripts/bench_latency.py --models all --platforms all

# Cross-platform parity
python scripts/bench_parity.py --model-path models/fusion_head.onnx
```

## Experiment Protocols

### R1: Embedding Choice vs Accuracy/Latency
**Hypothesis**: CLIP ViT-L/14 + e5-small yields best AUC-per-ms on-device.

**Variables**:
- Text encoders: {BERT-base, e5-small-v2, MiniLM}
- Image encoders: {CLIP ViT-B/32, CLIP ViT-L/14, EfficientNet-B0}

**Metrics**: AUC, F1@τ, throughput (img/s), p50/p95 latency (ms), model size (MB)

### R2: Fusion Architecture Ablations  
**Hypothesis**: MLP (text ⊕ image ⊕ delta features) outperforms any single-modality head.

**Variables**:
- Architecture: {Logistic Regression, 2-layer MLP, 3-layer MLP}
- Features: {text+image only, +deltas, +structural}
- Calibration: {none, Platt scaling, Isotonic regression}

**Metrics**: AUC, ECE, Brier score, F1@τ

### R3: Domain Adaptation
**Hypothesis**: LoRA adapters on text or image encoders boost in-domain F1 by ≥10% over zero-shot CLIP.

**Variables**:
- Adaptation: {none, head-only, LoRA-text, LoRA-image, LoRA-both}
- Domain: {general, product-photos, portraits, landscapes}

**Metrics**: ΔAUC/ΔF1 (in-domain and out-of-domain), generalization gap

### R4: Preference Learning for Ranking
**Hypothesis**: Pairwise preference loss improves top-k selection for multiple candidates.

**Variables**:
- Loss: {pointwise BCE, pairwise logistic, listwise}
- Ranking metrics: {NDCG@5, NDCG@10, Kendall's τ}

### R5: Robustness & Safety
**Hypothesis**: Score distributions shift under adversarial inputs; conformal prediction can keep error ≤ ε.

**Variables**:
- Corruption types: {over-saturation, copy-paste, unrelated instruction}
- Safety: {baseline, conformal prediction, uncertainty thresholding}

**Metrics**: Coverage, error @ fixed coverage, AUROC under corruptions

### R6: Real-time Batching & Quantization
**Hypothesis**: Adaptive micro-batching + INT8 head gives ≥1.8× throughput at ≤1 pt AUC drop.

**Variables**:
- Batching: {no batching, fixed batch, adaptive micro-batching}
- Quantization: {fp32, fp16, int8}
- Platforms: {M2, Android mid-range, WebGPU}

**Metrics**: Throughput, tail latency, AUC delta

### R7: Cross-platform Inference Parity
**Hypothesis**: ONNX Runtime WebGPU matches Core ML fp16 within ±0.5 AUC and ±10% latency.

**Variables**:
- Backends: {Core ML fp16, ONNX WebGPU, ONNX CPU, TFLite}
- Batch sizes: {1, 4, 8, 16}

**Metrics**: ΔAUC, Δlatency, operator fallbacks count

## Results Organization

### Experiment Results
Each experiment generates:
- `results/metrics.json`: Quantitative metrics
- `results/config.yaml`: Exact configuration used
- `results/model_hash.txt`: Model fingerprint
- `plots/performance.png`: Performance visualizations
- `Findings.md`: Summary of findings

### Aggregated Results
- `results/summary_table.csv`: All experiments in comparable format
- `plots/pareto_front.png`: Accuracy vs latency trade-offs
- `plots/calibration_curves.png`: Calibration across methods

## Reproducibility

### Determinism
- Fixed seeds for all random processes
- Package version logging
- Model hash verification
- Deterministic CUDA operations where possible

### Configuration-First
Every experiment is fully specified by YAML configuration:
```yaml
experiment:
  name: "r1_bert_clip_b32"
  description: "BERT-base + CLIP ViT-B/32 baseline"
  
models:
  text_encoder:
    name: "bert-base-uncased"
    pooling: "mean"
  image_encoder:
    name: "open_clip/ViT-B-32"
    pretrained: "laion2b_s34b_b79k"
    
training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 10
  seeds: [42, 43, 44]
  
evaluation:
  metrics: ["auc", "f1", "latency_p50", "latency_p95", "throughput"]
  platforms: ["cpu", "cuda", "mps"]
```

### Artifacts
All experiment artifacts are tracked:
- Model checkpoints (ONNX/Core ML)
- Training logs
- Evaluation results
- Visualization plots
- Configuration files

## Analysis Tools

### Result Comparison
```bash
# Compare two experiments
python scripts/compare_experiments.py --exp1 r1_bert_clip_b32 --exp2 r1_e5_clip_l14

# Generate pareto front
python scripts/generate_pareto.py --suite r1_embeddings --metric auc --metric latency_p95
```

### Statistical Analysis
```bash
# Run significance tests
python scripts/statistical_test.py --experiments r1_* --metric auc --test paired_ttest

# Generate confidence intervals
python scripts/confidence_intervals.py --results experiments/r1_embeddings/results/
```

## Documentation

Each experiment includes:
1. **Hypothesis**: Clear research question
2. **Method**: Experimental setup
3. **Metrics**: Evaluation criteria  
4. **Results**: Quantitative findings
5. **Discussion**: Interpretation and implications
6. **Artifacts**: All generated files

## Contributing New Experiments

1. Create experiment directory: `experiments/r8_new_research/`
2. Add configuration files in `configs/`
3. Implement experiment logic in `src/experiments/r8_new_research.py`
4. Add evaluation metrics if needed
5. Update this README with protocol

## Best Practices

- Use consistent naming conventions
- Include baseline comparisons
- Run multiple seeds for statistical significance
- Document all hyperparameters
- Store raw results and processed summaries
- Include negative controls where appropriate
- Verify cross-platform reproducibility
