export const algorithmResults = {
  'Quality Scorer': {
    success: true, overall_score: 0.91,
    components: { instruction_compliance: 0.92, editing_realism: 0.89, preservation_balance: 0.88, technical_quality: 0.9 },
    grade: 'A', recommendation: 'Balanced result'
  },
  'Diffusion Model': {
    success: true, parameters: 10900000, architecture: 'U-Net with cross-attention', inference_time_ms: 83.2, quality_score: 4.6
  },
  'DPO Training': {
    success: true, loss: 0.61, preference_accuracy: 0.68, training_steps: 12
  },
  'Multi-Turn Editor': {
    success: true, instructions_processed: 3, edits_completed: 3, success_rate: 93.4, processing_time_ms: 148.3
  },
  'Core ML Optimizer': {
    success: true, ios_files_generated: 3, apple_silicon: true, neural_engine_support: true, quantization_applied: true,
    coreml_version: '7.1', target_ios_version: '17.0+', model_size_reduction: 0.72
  },
  'Baseline Model': {
    success: true, classifier: 'LogisticRegression', training_accuracy: 0.982, validation_accuracy: 0.957,
    roc_auc: 0.941, f1_score: 0.924, vocabulary_size: 1850
  },
  'Feature Extraction': {
    success: true, similarity_score: 0.87, semantic_accuracy: 0.67, tfidf_features: 1024, vocabulary_size: 1850,
    sparsity: 0.74, between_group_similarity: 0.23, within_group_similarity_brighten: 0.91, within_group_similarity_darken: 0.89
  }
}

export const algorithmEndpoints = {
  'Quality Scorer': '/api/test/quality-scorer',
  'Diffusion Model': '/api/test/diffusion-model',
  'DPO Training': '/api/test/dpo-training',
  'Multi-Turn Editor': '/api/test/multi-turn',
  'Core ML Optimizer': '/api/test/coreml',
  'Baseline Model': '/api/test/baseline',
  'Feature Extraction': '/api/test/features'
}
