//
//  EditJudgeViewModel.swift
//  PicoTuri-EditJudge iOS Demo
//
//  ViewModel for managing edit quality assessment
//

import SwiftUI
import Combine

struct ComponentScore: Identifiable {
    let id = UUID()
    let name: String
    let score: Double
    let weight: Double
}

class EditJudgeViewModel: ObservableObject {
    @Published var overallScore: Double = 0.0
    @Published var grade: String = ""
    @Published var recommendation: String = ""
    @Published var componentScores: [ComponentScore] = []
    @Published var hasResults: Bool = false
    
    func analyzeImages(original: UIImage, edited: UIImage, instruction: String) {
        // Simulate quality assessment
        // In a real app, this would use Core ML model
        
        let instructionCompliance = Double.random(in: 0.6...0.95)
        let editingRealism = Double.random(in: 0.5...0.9)
        let preservationBalance = Double.random(in: 0.7...0.95)
        let technicalQuality = Double.random(in: 0.6...0.9)
        
        // Calculate weighted overall score
        let overall = (
            instructionCompliance * 0.40 +
            editingRealism * 0.25 +
            preservationBalance * 0.20 +
            technicalQuality * 0.15
        )
        
        withAnimation {
            self.overallScore = overall
            self.componentScores = [
                ComponentScore(name: "Instruction Compliance", score: instructionCompliance, weight: 0.40),
                ComponentScore(name: "Editing Realism", score: editingRealism, weight: 0.25),
                ComponentScore(name: "Preservation Balance", score: preservationBalance, weight: 0.20),
                ComponentScore(name: "Technical Quality", score: technicalQuality, weight: 0.15)
            ]
            
            // Determine grade
            if overall >= 0.85 {
                self.grade = "Excellent"
                self.recommendation = "Outstanding edit quality! The image meets all quality criteria."
            } else if overall >= 0.70 {
                self.grade = "Good"
                self.recommendation = "Good edit quality with minor improvements possible."
            } else if overall >= 0.50 {
                self.grade = "Fair"
                self.recommendation = "Acceptable quality but could benefit from refinement."
            } else {
                self.grade = "Poor"
                self.recommendation = "Significant improvements needed to meet quality standards."
            }
            
            self.hasResults = true
        }
    }
    
    func reset() {
        withAnimation {
            self.overallScore = 0.0
            self.grade = ""
            self.recommendation = ""
            self.componentScores = []
            self.hasResults = false
        }
    }
}
