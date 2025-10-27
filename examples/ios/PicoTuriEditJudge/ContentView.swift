//
//  ContentView.swift
//  PicoTuri-EditJudge iOS Demo
//
//  Main view with image editing quality assessment
//

import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var viewModel = EditJudgeViewModel()
    @State private var selectedImage: UIImage?
    @State private var editedImage: UIImage?
    @State private var instruction: String = ""
    @State private var showImagePicker = false
    @State private var showEditedPicker = false
    @State private var isAnalyzing = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    headerView
                    
                    // Image Selection
                    imageSelectionView
                    
                    // Instruction Input
                    instructionView
                    
                    // Analyze Button
                    analyzeButton
                    
                    // Results
                    if viewModel.hasResults {
                        resultsView
                    }
                    
                    // Algorithm Status
                    algorithmStatusView
                }
                .padding()
            }
            .navigationTitle("PicoTuri EditJudge")
            .navigationBarTitleDisplayMode(.large)
        }
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(image: $selectedImage)
        }
        .sheet(isPresented: $showEditedPicker) {
            ImagePicker(image: $editedImage)
        }
    }
    
    // MARK: - Header View
    private var headerView: some View {
        VStack(spacing: 8) {
            Image(systemName: "photo.badge.checkmark.fill")
                .font(.system(size: 60))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            
            Text("Image Edit Quality Assessment")
                .font(.headline)
                .foregroundColor(.secondary)
        }
        .padding(.vertical)
    }
    
    // MARK: - Image Selection View
    private var imageSelectionView: some View {
        VStack(spacing: 15) {
            HStack(spacing: 15) {
                // Original Image
                VStack {
                    Text("Original")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 150, height: 150)
                            .clipShape(RoundedRectangle(cornerRadius: 15))
                            .overlay(
                                RoundedRectangle(cornerRadius: 15)
                                    .stroke(Color.blue, lineWidth: 2)
                            )
                    } else {
                        RoundedRectangle(cornerRadius: 15)
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: 150, height: 150)
                            .overlay(
                                VStack {
                                    Image(systemName: "photo")
                                        .font(.largeTitle)
                                        .foregroundColor(.gray)
                                    Text("Tap to select")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                            )
                    }
                }
                .onTapGesture {
                    showImagePicker = true
                }
                
                // Edited Image
                VStack {
                    Text("Edited")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if let image = editedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 150, height: 150)
                            .clipShape(RoundedRectangle(cornerRadius: 15))
                            .overlay(
                                RoundedRectangle(cornerRadius: 15)
                                    .stroke(Color.purple, lineWidth: 2)
                            )
                    } else {
                        RoundedRectangle(cornerRadius: 15)
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: 150, height: 150)
                            .overlay(
                                VStack {
                                    Image(systemName: "photo.badge.plus")
                                        .font(.largeTitle)
                                        .foregroundColor(.gray)
                                    Text("Tap to select")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                            )
                    }
                }
                .onTapGesture {
                    showEditedPicker = true
                }
            }
        }
    }
    
    // MARK: - Instruction View
    private var instructionView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Edit Instruction")
                .font(.headline)
            
            TextField("e.g., 'brighten the image'", text: $instruction)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal, 12)
                .padding(.vertical, 12)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
        }
    }
    
    // MARK: - Analyze Button
    private var analyzeButton: some View {
        Button(action: analyzeImages) {
            HStack {
                if isAnalyzing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Image(systemName: "sparkles")
                }
                Text(isAnalyzing ? "Analyzing..." : "Analyze Quality")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                LinearGradient(
                    colors: canAnalyze ? [.blue, .purple] : [.gray],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .foregroundColor(.white)
            .cornerRadius(15)
        }
        .disabled(!canAnalyze || isAnalyzing)
    }
    
    // MARK: - Results View
    private var resultsView: some View {
        VStack(spacing: 15) {
            // Overall Score
            VStack(spacing: 10) {
                Text("Overall Quality Score")
                    .font(.headline)
                
                ZStack {
                    Circle()
                        .stroke(Color.gray.opacity(0.2), lineWidth: 20)
                        .frame(width: 150, height: 150)
                    
                    Circle()
                        .trim(from: 0, to: viewModel.overallScore)
                        .stroke(
                            LinearGradient(
                                colors: [.green, .blue, .purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            style: StrokeStyle(lineWidth: 20, lineCap: .round)
                        )
                        .frame(width: 150, height: 150)
                        .rotationEffect(.degrees(-90))
                        .animation(.spring(), value: viewModel.overallScore)
                    
                    VStack {
                        Text("\(Int(viewModel.overallScore * 100))")
                            .font(.system(size: 40, weight: .bold))
                        Text("/ 100")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Text(viewModel.grade)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(gradeColor)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)
                    .background(gradeColor.opacity(0.2))
                    .cornerRadius(10)
            }
            .padding()
            .background(Color.gray.opacity(0.05))
            .cornerRadius(15)
            
            // Component Scores
            VStack(alignment: .leading, spacing: 12) {
                Text("Component Breakdown")
                    .font(.headline)
                    .padding(.bottom, 5)
                
                ForEach(viewModel.componentScores, id: \.name) { component in
                    ComponentScoreRow(
                        name: component.name,
                        score: component.score,
                        weight: component.weight
                    )
                }
            }
            .padding()
            .background(Color.gray.opacity(0.05))
            .cornerRadius(15)
            
            // Recommendation
            if !viewModel.recommendation.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Recommendation", systemImage: "lightbulb.fill")
                        .font(.headline)
                        .foregroundColor(.orange)
                    
                    Text(viewModel.recommendation)
                        .font(.body)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(15)
            }
        }
    }
    
    // MARK: - Algorithm Status View
    private var algorithmStatusView: some View {
        VStack(spacing: 12) {
            Text("Algorithm Status")
                .font(.headline)
            
            HStack(spacing: 20) {
                StatusBadge(title: "Quality Scorer", status: .ready)
                StatusBadge(title: "Neural Engine", status: .active)
                StatusBadge(title: "Core ML", status: .ready)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(15)
    }
    
    // MARK: - Computed Properties
    private var canAnalyze: Bool {
        selectedImage != nil && editedImage != nil && !instruction.isEmpty
    }
    
    private var gradeColor: Color {
        switch viewModel.grade {
        case "Excellent": return .green
        case "Good": return .blue
        case "Fair": return .orange
        default: return .red
        }
    }
    
    // MARK: - Methods
    private func analyzeImages() {
        guard let original = selectedImage,
              let edited = editedImage else { return }
        
        isAnalyzing = true
        
        // Simulate analysis (in real app, this would call Core ML model)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            viewModel.analyzeImages(original: original, edited: edited, instruction: instruction)
            isAnalyzing = false
        }
    }
}

// MARK: - Component Score Row
struct ComponentScoreRow: View {
    let name: String
    let score: Double
    let weight: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text("\(Int(score * 100))%")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.blue)
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 8)
                    
                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            LinearGradient(
                                colors: [.blue, .purple],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * score, height: 8)
                        .animation(.spring(), value: score)
                }
            }
            .frame(height: 8)
            
            Text("Weight: \(Int(weight * 100))%")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Status Badge
struct StatusBadge: View {
    enum Status {
        case ready, active, offline
        
        var color: Color {
            switch self {
            case .ready: return .green
            case .active: return .blue
            case .offline: return .gray
            }
        }
        
        var icon: String {
            switch self {
            case .ready: return "checkmark.circle.fill"
            case .active: return "bolt.circle.fill"
            case .offline: return "xmark.circle.fill"
            }
        }
    }
    
    let title: String
    let status: Status
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: status.icon)
                .font(.title2)
                .foregroundColor(status.color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Image Picker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.image = image
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
