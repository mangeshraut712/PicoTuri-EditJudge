import CoreML
import Foundation
import SwiftUI

@MainActor
final class ModelController: ObservableObject {
    @Published var instruction: String = "Brighten the living room photo"
    @Published var imageSimilarity: Double = 0.82
    @Published private(set) var score: Double = 0.0
    @Published private(set) var threshold: Double = 0.5
    @Published private(set) var statusMessage: String = "Tap Score to evaluate."

    private var coreMLModel: MLModel?

    func loadResources() async {
        // Load Core ML model and thresholds
        await loadCoreMLModel()
        await loadThresholds()
    }

    private func loadCoreMLModel() async {
        do {
            if let modelURL = Bundle.main.url(forResource: "PicoTuriEditJudge", withExtension: "mlmodelc") {
                coreMLModel = try MLModel(contentsOf: modelURL)
                statusMessage = "Loaded Core ML model."
            } else {
                statusMessage = "Core ML model not found in bundle."
            }
        } catch {
            statusMessage = "Failed to load Core ML model: \\(error.localizedDescription)"
        }
    }

    private func loadThresholds() async {
        if let thresholdURL = Bundle.main.url(forResource: "thresholds", withExtension: "json") {
            do {
                let data = try Data(contentsOf: thresholdURL)
                let decoded = try JSONDecoder().decode(ThresholdConfig.self, from: data)
                threshold = decoded.scoreThreshold
                statusMessage = "Loaded thresholds from bundle."
            } catch {
                statusMessage = "Failed to load thresholds: \\(error.localizedDescription)"
            }
        }
    }

    func scoreEdit() {
        guard let model = coreMLModel else {
            // Fallback to placeholder scoring if model not loaded
            fallbackScoreEdit()
            return
        }

        do {
            // Prepare input for Core ML model
            let input = PicoTuriEditJudgeInput(
                instruction: instruction,
                image_similarity: imageSimilarity
            )

            // Make prediction
            let prediction = try model.prediction(from: input)

            // Extract score from prediction
            if let scoreOutput = prediction.featureValue(for: "score")?.doubleValue {
                score = scoreOutput
                statusMessage = score >= threshold ? "Passes baseline threshold." : "Falls below baseline threshold."
            } else {
                fallbackScoreEdit()
            }
        } catch {
            statusMessage = "Prediction failed: \\(error.localizedDescription)"
            fallbackScoreEdit()
        }
    }

    private func fallbackScoreEdit() {
        // Placeholder scoring using simple heuristics when Core ML model unavailable
        let textBoost = instruction.count > 0 ? 0.1 : 0.0
        score = min(1.0, max(0.0, imageSimilarity * 0.8 + textBoost))
        statusMessage = score >= threshold ? "Passes baseline threshold (fallback)." : "Falls below baseline threshold (fallback)."
    }
}

private struct ThresholdConfig: Decodable {
    let scoreThreshold: Double

    enum CodingKeys: String, CodingKey {
        case scoreThreshold = "score_threshold"
    }
}
