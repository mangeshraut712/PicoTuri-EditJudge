# 📁 PicoTuri-EditJudge Project Structure

## 🗂️ Directory Organization

```
PicoTuri-EditJudge/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # Apache 2.0 license
├── 📄 CITATION.cff                 # Citation information
├── 📄 CODE_OF_CONDUCT.md           # Code of conduct
├── 📄 CONTRIBUTING.md              # Contributing guidelines
├── 📄 requirements-dev.txt         # Python dependencies
├── 📄 verify_all.sh                # Quick verification script
│
├── 📂 src/                         # Main source code
│   ├── 📂 algorithms/              # Core algorithms
│   │   ├── quality_scorer.py      # 4-component quality scorer
│   │   ├── diffusion_model.py     # U-Net diffusion model
│   │   ├── dpo_training.py        # DPO training
│   │   ├── multi_turn_editor.py   # Multi-turn editor
│   │   ├── coreml_optimizer.py    # Core ML optimization
│   │   └── deep_dive.py           # Advanced algorithms
│   │
│   ├── 📂 train/                   # Training modules
│   │   └── baseline.py            # Baseline training
│   │
│   ├── 📂 export/                  # Model export utilities
│   │   └── coreml_export.py       # Core ML export
│   │
│   ├── 📂 features_image/          # Image feature extraction
│   │   └── similarity.py          # Image similarity
│   │
│   ├── 📂 features_text/           # Text feature extraction
│   │   └── tfidf.py               # TF-IDF vectorizer
│   │
│   ├── 📂 fuse/                    # Feature fusion
│   │   └── feature_joiner.py      # Feature joining
│   │
│   ├── 📂 gui/                     # GUI components
│   │   └── dashboard.py           # Performance dashboard
│   │
│   └── 📂 models/                  # Model definitions
│       └── simple_instruction_processor.py
│
├── 📂 tests/                       # Test suite
│   ├── test_all_algorithms.py     # Comprehensive tests
│   └── test_smoke.py              # Smoke tests
│
├── 📂 docs/                        # Documentation
│   ├── ALGORITHM_VERIFICATION_REPORT.md
│   ├── QUICK_START_GUIDE.md
│   ├── FIXES_SUMMARY.md
│   └── LINTING_FIXES_REPORT.md
│
├── 📂 assets/                      # Project assets
│   ├── 📂 images/                  # Sample images
│   │   └── test.jpg
│   └── 📂 charts/                  # Generated charts
│       ├── algorithm_comparison.png
│       ├── dataset_table.png
│       ├── metrics_dashboard.png
│       └── predictions_bar.png
│
├── 📂 data/                        # Data files
│   ├── sample_dataset.csv         # Sample dataset
│   ├── baseline.joblib            # Trained baseline model
│   └── 📂 manifests/               # Dataset manifests
│
├── 📂 examples/                    # Example implementations
│   ├── 📂 ios/                     # iOS SwiftUI examples
│   │   └── EditJudgeDemo/
│   └── 📂 c_demo/                  # C integration examples
│
├── 📂 scripts/                     # Utility scripts
│   ├── setup_environment.py       # Environment setup
│   ├── download_data.py           # Data download
│   └── pico_banana_setup.py       # Dataset setup
│
├── 📂 tools/                       # Development tools
│   ├── download_pico_banana_dataset.py
│   ├── map_openimage_url_to_local.py
│   └── optimize_for_coreml.py
│
├── 📂 configs/                     # Configuration files
│   ├── 📂 envs/                    # Environment configs
│   └── 📂 models/                  # Model configs
│
└── 📂 notebooks/                   # Jupyter notebooks
    └── exploration.ipynb          # Data exploration
```

## 🎯 Key Directories

### Source Code (`src/`)
- **algorithms/** - Core ML algorithms (quality scorer, diffusion, DPO, etc.)
- **train/** - Training pipelines
- **export/** - Model export utilities
- **features_image/** - Image feature extraction
- **features_text/** - Text feature extraction
- **fuse/** - Feature fusion modules
- **gui/** - Dashboard and visualization
- **models/** - Model definitions

### Tests (`tests/`)
- **test_all_algorithms.py** - Comprehensive algorithm testing
- **test_smoke.py** - Quick smoke tests

### Documentation (`docs/`)
- Algorithm verification reports
- Quick start guides
- Fix summaries
- Linting reports

### Assets (`assets/`)
- **images/** - Sample images for testing
- **charts/** - Generated performance charts

### Data (`data/`)
- Sample datasets
- Trained models
- Dataset manifests

### Examples (`examples/`)
- **ios/** - iOS SwiftUI demo app
- **c_demo/** - C integration examples

### Scripts (`scripts/`)
- Environment setup
- Data download utilities
- Dataset preparation

### Tools (`tools/`)
- Development utilities
- Optimization tools
- Dataset management

## 🔧 Configuration Files

- **.flake8** - Flake8 linting configuration
- **.gitignore** - Git ignore patterns
- **.python-version** - Python version specification
- **requirements-dev.txt** - Python dependencies

## 📊 Generated Files

The following directories contain generated/output files:
- `assets/charts/` - Performance charts (PNG files)
- `data/` - Trained models and datasets

## 🚀 Quick Navigation

### Run Tests
```bash
python tests/test_all_algorithms.py
# or
./verify_all.sh
```

### Train Model
```bash
python -m src.train.baseline --pairs data/sample_dataset.csv --model-path data/baseline.joblib
```

### Generate Dashboard
```bash
python src/gui/dashboard.py
```

### Run Individual Algorithms
```bash
python src/algorithms/quality_scorer.py
python src/algorithms/diffusion_model.py
python -m src.algorithms.dpo_training
python src/algorithms/multi_turn_editor.py
python -m src.algorithms.coreml_optimizer
```

## 📝 Notes

- All source code is in `src/`
- All tests are in `tests/`
- All documentation is in `docs/`
- All assets (images, charts) are in `assets/`
- Configuration files are in the root or `configs/`
- Examples and demos are in `examples/`

## 🔍 File Locations

| File Type | Location |
|-----------|----------|
| Source Code | `src/` |
| Tests | `tests/` |
| Documentation | `docs/` |
| Images | `assets/images/` |
| Charts | `assets/charts/` |
| Data | `data/` |
| Examples | `examples/` |
| Scripts | `scripts/` |
| Tools | `tools/` |
| Configs | `configs/` |

---

**Last Updated:** October 27, 2025  
**Structure Version:** 2.0 (Cleaned and Organized)
