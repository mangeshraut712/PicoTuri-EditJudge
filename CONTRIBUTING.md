# Contributing to PicoTuri-EditJudge

Thanks for your interest in improving PicoTuri-EditJudge! We welcome thoughtful contributions that strengthen the baseline verifier, improve documentation, polish the on-device demos, or help us move beyond Turi Create in 2025.

## Ground Rules

- Respect the [Code of Conduct](CODE_OF_CONDUCT.md).
- Use small, focused pull requests with clear commit messages.
- Document dataset provenance and licensing for every contribution. Pico-Banana-400K and derivative annotations remain **CC BY-NC-ND 4.0**; do not upload raw media or large manifest files.
- Keep the code base Apple-friendly: strive for Core ML compatibility, low-latency designs, and Rosetta-friendly tooling until we transition to native arm64 paths.

## Development Workflow

1. **Fork & Clone.**
   ```bash
   gh repo fork apple-ondevice-research/PicoTuri-EditJudge --clone
   cd PicoTuri-EditJudge
   ```
2. **Create an environment** (Python 3.8 x86_64).
   ```bash
   python3.8 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
3. **Run tests and linters** before every push.
   ```bash
   pytest
   flake8 src tests
   mypy src
   ```
4. **Add documentation** for new features. Update `README.md`, `docs/`, and configs as appropriate.
5. **Open a pull request** following the template. Include screenshots/GIFs for UI changes and Core ML benchmarks when relevant.

## Commit & PR Style

- Prefix commits with conventional tags (`feat:`, `fix:`, `docs:`, `ci:`, `test:`, `chore:`).
- Reference related issues in PR descriptions.
- Include reproduction steps and expected outcomes when fixing bugs.
- Add or update tests covering your changes.

## Testing Guidance

- `tests/test_smoke.py` ensures our sample manifest and data contracts continue to work.
- Use additional unit tests under `tests/` for new modules.
- When touching Core ML export or SwiftUI code, add integration notes or sample projects for local validation.

## Adding Data or Models

- Store only lightweight manifests (≤50 rows) and metrics in the repository.
- Keep larger assets in private storage; document access steps in `data/scripts/`.
- Update `configs/models/` whenever thresholds, metadata, or exported bundles change.

## Releasing Updates

- Tag releases as `vMAJOR.MINOR.PATCH`.
- Update `CITATION.cff`, `README`, and roadmap milestones when shipping notable features.
- Include model cards and dataset cards for every public release.

Thank you for helping us build a responsible, on-device-first edit judge! ✨
