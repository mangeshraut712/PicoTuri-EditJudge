# EditJudgeDemo (SwiftUI)

Placeholder SwiftUI application that will host the on-device PicoTuri-EditJudge experience once the Core ML export is available.

## Getting Started

1. Launch Xcode 15+.
2. Create a new SwiftUI App named **EditJudgeDemo** targeting iOS 17.
3. Replace the generated `ContentView.swift` and `EditJudgeDemoApp.swift` with the files in this folder.
4. Drag the exported model bundle (e.g., `EditJudge.mlpackage`) into the project.
5. Update the `ModelController` to point at the threshold file in `configs/models/thresholds.json`.

## Roadmap Hooks

- Add live scoring once the Core ML export is ready.
- Surface energy/latency measurements using `MetricKit`.
- Expose router decisions (Core Image vs. generative pipeline) in the UI.
