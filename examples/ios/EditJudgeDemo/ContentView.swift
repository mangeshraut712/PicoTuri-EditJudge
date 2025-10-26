import SwiftUI

struct ContentView: View {
    @ObservedObject var model: ModelController

    var body: some View {
        NavigationStack {
            Form {
                Section("Instruction") {
                    TextField("Describe the edit", text: $model.instruction)
                }

                Section("Image Similarity") {
                    Slider(value: $model.imageSimilarity, in: 0...1)
                    Text(String(format: "%.2f", model.imageSimilarity))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Score") {
                    Toggle("Good edit?", isOn: Binding(get: {
                        model.score >= model.threshold
                    }, set: { _ in }))
                    .tint(.accentColor)

                    Text(model.statusMessage)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Debug") {
                    Text("Score: \(String(format: \"%.3f\", model.score))")
                    Text("Threshold: \(String(format: \"%.3f\", model.threshold))")
                }
            }
            .navigationTitle("PicoTuri Edit Judge")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Score") {
                        model.scoreEdit()
                    }
                }
            }
        }
        .task {
            await model.loadResources()
        }
    }
}

#Preview {
    ContentView(model: ModelController())
}
