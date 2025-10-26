# C Demo Placeholder

This folder will eventually contain a minimal `libeditjudge` consumer showing how to link against the C ABI exposed in Milestone 0.6.

Planned contents:

- `include/editjudge.h` – C header mirroring the Swift interface.
- `src/main.c` – Simple CLI that scores a manifest of edits.
- `CMakeLists.txt` – Build system hooks for macOS and Linux.

For now, use the SwiftUI demo under `examples/ios/EditJudgeDemo` to validate Core ML exports.
