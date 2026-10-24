const numericFields = {
  'Quality Scorer': ['overall_score'],
  'Diffusion Model': ['parameters', 'inference_time_ms', 'quality_score'],
  'Multi-Turn Editor': ['instructions_processed', 'edits_completed', 'processing_time_ms', 'success_rate'],
  'DPO Training': ['loss', 'preference_accuracy', 'training_steps'],
  'Core ML Optimizer': ['ios_files_generated', 'model_size_reduction'],
  'Baseline Model': ['training_accuracy', 'validation_accuracy', 'f1_score', 'roc_auc', 'vocabulary_size'],
  'Feature Extraction': ['tfidf_features', 'similarity_score', 'semantic_accuracy', 'vocabulary_size', 'sparsity']
}

const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value)
const inRange = (value, min, max) => Number.isFinite(value) && value >= min && value <= max

// Validate data before formatting it or passing it to chart components.
export const isValidAlgorithmResult = (name, result) => {
  if (!Object.hasOwn(numericFields, name) || !isRecord(result) || result.success !== true) return false
  if (!numericFields[name].every(key => Number.isFinite(result[key]))) return false
  const optionalNumbers = ['average_confidence', 'failed_edits', 'between_group_similarity',
    'within_group_similarity_brighten', 'within_group_similarity_darken']
  if (optionalNumbers.some(key => result[key] != null && !Number.isFinite(result[key]))) return false
  const fractions = ['overall_score', 'average_confidence', 'model_size_reduction',
    'preference_accuracy', 'training_accuracy', 'validation_accuracy', 'f1_score', 'roc_auc',
    'semantic_accuracy', 'sparsity']
  if (fractions.some(key => result[key] != null && !inRange(result[key], 0, 1))) return false
  const similarities = ['similarity_score', 'between_group_similarity',
    'within_group_similarity_brighten', 'within_group_similarity_darken']
  if (similarities.some(key => result[key] != null && !inRange(result[key], -1, 1))) return false
  const counts = ['parameters', 'instructions_processed', 'edits_completed', 'failed_edits',
    'training_steps', 'ios_files_generated', 'vocabulary_size', 'tfidf_features']
  if (counts.some(key => result[key] != null && (!Number.isInteger(result[key]) || result[key] < 0))) return false
  if (['inference_time_ms', 'processing_time_ms', 'loss'].some(key =>
    result[key] != null && (!Number.isFinite(result[key]) || result[key] < 0))) return false
  if (result.quality_score != null && !inRange(result.quality_score, 0, 5)) return false
  if (result.success_rate != null && !inRange(result.success_rate, 0, 100)) return false
  const booleans = ['apple_silicon', 'neural_engine_support', 'quantization_applied',
    'supports_text_to_image', 'supports_image_to_image']
  if (booleans.some(key => result[key] != null && typeof result[key] !== 'boolean')) return false
  const optionalStrings = ['grade', 'recommendation', 'architecture', 'target_ios_version',
    'coreml_version', 'classifier', 'embedding_model', 'method']
  if (optionalStrings.some(key => result[key] != null && typeof result[key] !== 'string')) return false
  if (name === 'Quality Scorer' || result.components !== undefined) {
    if (!isRecord(result.components) || Object.keys(result.components).length === 0 ||
        !Object.values(result.components).every(value => inRange(value, 0, 1))) return false
  }
  if (result.training_history !== undefined) {
    const history = result.training_history
    if (!isRecord(history) || !Array.isArray(history.loss) || !Array.isArray(history.accuracy) ||
        history.loss.length === 0 || history.loss.length !== history.accuracy.length ||
        !history.loss.every(value => Number.isFinite(value) && value >= 0) ||
        !history.accuracy.every(value => inRange(value, 0, 100))) return false
  }
  return true
}
