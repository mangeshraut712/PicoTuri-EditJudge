import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts'

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

const ChartVisualization = ({ algorithm, result }) => {
  if (!result) return <div className="text-gray-400">No data available</div>
  if (!isValidAlgorithmResult(algorithm?.name, result)) {
    return <div role="alert" className="text-red-300">No valid algorithm results available.</div>
  }

  // Quality Scorer - Component breakdown
  if (algorithm.name === 'Quality Scorer') {
    const chartData = Object.entries(result.components).map(([key, value]) => ({
      name: key.split('_').filter(Boolean).map(w => w[0].toUpperCase() + w.slice(1)).join(' '),
      value: parseFloat((value * 100).toFixed(1))
    }))

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Radar Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Component Analysis</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={chartData}>
                <PolarGrid stroke="#ffffff20" />
                <PolarAngleAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <Radar
                  name="Score"
                  dataKey="value"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.3}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Bar Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Score Breakdown</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="value" fill="#8884d8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="text-center p-6 glass rounded-xl">
          <div className="text-4xl font-bold text-purple-400 mb-2">
            {(result.overall_score * 100).toFixed(1)}%
          </div>
          <div className="text-xl text-white mb-1">Overall Score</div>
          <div className="text-sm text-gray-400">
            Grade: {result.grade || 'Not reported'}{result.recommendation && ` | ${result.recommendation}`}
          </div>
        </div>
      </div>
    )
  }

  // Diffusion Model - Architecture info with Line Chart
  if (algorithm.name === 'Diffusion Model') {
    const architectureData = [
      { name: 'Parameter Count', value: result.parameters / 1000000, unit: 'M', color: '#8884d8' },
      { name: 'Inference Time', value: result.inference_time_ms, unit: 'ms', color: '#82ca9d' },
      { name: 'Quality Score', value: result.quality_score, unit: '/5', color: '#ffc658' }
    ]

    const performanceData = [{
      sample: 'Reported result',
      latency: result.inference_time_ms,
      quality: result.quality_score
    }]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Architecture Cards */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Model Architecture</h3>
            <div className="space-y-3">
              {architectureData.map((item, index) => (
                <div key={index} className="glass p-4 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">{item.name}</span>
                    <span className="text-lg font-bold" style={{ color: item.color }}>
                      {item.value.toFixed(1)} {item.unit}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Line Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Reported Performance</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="sample" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis yAxisId="latency" unit=" ms" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis yAxisId="quality" orientation="right" domain={[0, 5]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Line
                  yAxisId="latency"
                  type="monotone"
                  dataKey="latency"
                  stroke="#8884d8"
                  strokeWidth={2}
                  name="Latency (ms)"
                  dot={{ r: 4 }}
                />
                <Line
                  yAxisId="quality"
                  type="monotone"
                  dataKey="quality"
                  stroke="#82ca9d"
                  strokeWidth={2}
                  name="Quality Score (/5)"
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="text-center p-6 glass rounded-xl">
          <div className="text-xl text-white mb-2">{result.architecture || 'Architecture not reported'}</div>
          <div className="text-sm text-gray-400">
            Text-to-image: {result.supports_text_to_image == null ? 'Not reported' : result.supports_text_to_image ? 'Supported' : 'Not supported'}
            {' | '}Image-to-image: {result.supports_image_to_image == null ? 'Not reported' : result.supports_image_to_image ? 'Supported' : 'Not supported'}
          </div>
        </div>
      </div>
    )
  }

  // Multi-Turn Editor - Session flow visualization
  if (algorithm.name === 'Multi-Turn Editor') {
    const sessionData = [
      { name: 'Completed', value: result.edits_completed, color: '#8b5cf6' },
      ...(result.failed_edits == null ? [] : [{ name: 'Failed', value: result.failed_edits, color: '#ef4444' }])
    ]

    const responseTimes = [{
      request: 'Session total',
      time: result.processing_time_ms
    }]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Session Flow Pie Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Session Flow</h3>
            {result.failed_edits == null && <p className="text-sm text-gray-400">Failed edit count was not reported.</p>}
            {sessionData.every(entry => entry.value === 0) && <p className="text-sm text-gray-400">No completed or failed edits reported.</p>}
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={sessionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {sessionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Performance Scatter Plot */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Session Processing Time</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={responseTimes}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="request" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="time"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Response Time (ms)"
                  dot={{ fill: '#10b981', r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-green-400">{result.instructions_processed}</div>
            <div className="text-sm text-gray-400">Instructions</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-blue-400">{result.edits_completed}</div>
            <div className="text-sm text-gray-400">Edits</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-purple-400">{Number.isFinite(result.average_confidence) ? `${(result.average_confidence * 100).toFixed(0)}%` : 'Not reported'}</div>
            <div className="text-sm text-gray-400">Confidence</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-orange-400">{(result.success_rate)?.toFixed(1)}%</div>
            <div className="text-sm text-gray-400">Success</div>
          </div>
        </div>
      </div>
    )
  }

  // DPO Training - Loss and accuracy over time
  if (algorithm.name === 'DPO Training') {
    const history = result.training_history
    const chartData = history
      ? history.loss.map((loss, index) => ({
        step: index + 1,
        loss,
        // Training history reports percentages; the final accuracy is a fraction.
        accuracy: history.accuracy[index] / 100
      }))
      : [{ step: result.training_steps, loss: result.loss, accuracy: result.preference_accuracy }]

    return (
      <div className="space-y-6">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
            <XAxis dataKey="step" tick={{ fill: '#9ca3af' }} />
            <YAxis yAxisId="left" tick={{ fill: '#9ca3af' }} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tick={{ fill: '#9ca3af' }} tickFormatter={value => `${(value * 100).toFixed(0)}%`} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px'
              }}
              formatter={(value, name) => [name === 'Accuracy' ? `${(value * 100).toFixed(1)}%` : value, name]}
            />
            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} name="Loss" />
            <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} name="Accuracy" />
          </LineChart>
        </ResponsiveContainer>

        {!history && <p className="text-sm text-gray-400">Only the final result was supplied; per-step training history is unavailable.</p>}
        <div className="grid grid-cols-3 gap-4">
          <div className="glass p-6 rounded-xl text-center">
            <div className="text-2xl font-bold text-red-400">
              {result.loss?.toFixed(4)}
            </div>
            <div className="text-sm text-gray-400">Final Loss</div>
          </div>
          <div className="glass p-6 rounded-xl text-center">
            <div className="text-2xl font-bold text-green-400">
              {(result.preference_accuracy * 100)?.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Preference Accuracy</div>
          </div>
          <div className="glass p-6 rounded-xl text-center">
            <div className="text-2xl font-bold text-blue-400">
              {result.training_steps}
            </div>
            <div className="text-sm text-gray-400">Training Steps</div>
          </div>
        </div>
      </div>
    )
  }

  // Core ML Optimizer - Optimization visualization
  if (algorithm.name === 'Core ML Optimizer') {
    const optimizationData = [
      { metric: 'Model Size (%)', before: 100, after: Number(((1 - result.model_size_reduction) * 100).toFixed(1)) }
    ]

    const featuresData = [
      { name: 'Apple Silicon', available: result.apple_silicon },
      { name: 'Neural Engine', available: result.neural_engine_support },
      { name: 'Quantization', available: result.quantization_applied }
    ]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Optimization Metrics Bar Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Optimization Results</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={optimizationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="before" fill="#ffffff40" name="Before" />
                <Bar dataKey="after" fill="#10b981" name="After" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Feature Support</h3>
            <dl className="space-y-3">
              {featuresData.map(feature => (
                <div key={feature.name} className="glass p-4 rounded-lg flex justify-between">
                  <dt className="text-gray-400">{feature.name}</dt>
                  <dd className="text-white">{feature.available == null ? 'Not reported' : feature.available ? 'Available' : 'Not available'}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-red-400">{result.ios_files_generated}</div>
            <div className="text-sm text-gray-400">Files Generated</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-green-400">{(result.model_size_reduction * 100).toFixed(0)}%</div>
            <div className="text-sm text-gray-400">Size Reduction</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-blue-400">{result.target_ios_version || 'Not reported'}</div>
            <div className="text-sm text-gray-400">Target Version</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-purple-400">{result.coreml_version || 'Not reported'}</div>
            <div className="text-sm text-gray-400">Core ML Version</div>
          </div>
        </div>
      </div>
    )
  }

  // Baseline Model - Performance scatter plot
  if (algorithm.name === 'Baseline Model') {
    const classifierName = result.classifier || 'Baseline Model'
    const modelComparisonData = [
      { model: classifierName, accuracy: result.training_accuracy, validation: result.validation_accuracy, f1: result.f1_score }
    ]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Model Comparison Scatter Plot */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Reported Model Performance</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={modelComparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                  formatter={(value, name) => [`${(value * 100).toFixed(2)}%`, name]}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10b981"
                  strokeWidth={3}
                  name="Training Accuracy"
                  dot={{ fill: '#10b981', r: 5 }}
                />
                <Line type="monotone" dataKey="validation" stroke="#8b5cf6" name="Validation Accuracy" />
                <Line
                  type="monotone"
                  dataKey="f1"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="F1-Score"
                  dot={{ fill: '#3b82f6', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Processing Pipeline</h3>
            <p className="text-sm text-gray-400">Per-stage processing times were not supplied.</p>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-green-400">{(result.training_accuracy ?? 0).toFixed(3)}</div>
            <div className="text-sm text-gray-400">Training Accuracy</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-blue-400">{(result.f1_score ?? 0).toFixed(3)}</div>
            <div className="text-sm text-gray-400">F1 Score</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-purple-400">{(result.roc_auc ?? 0).toFixed(3)}</div>
            <div className="text-sm text-gray-400">ROC AUC</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-orange-400">{result.vocabulary_size ?? 0}</div>
            <div className="text-sm text-gray-400">Vocabulary Size</div>
          </div>
        </div>
      </div>
    )
  }

  // Feature Extraction - Similarity matrix visualization
  if (algorithm.name === 'Feature Extraction') {
    const similarityData = [
      { pair: 'brighten-darken', similarity: result.between_group_similarity ?? null },
      { pair: 'brighten-increase', similarity: result.within_group_similarity_brighten ?? null },
      { pair: 'darken-decrease', similarity: result.within_group_similarity_darken ?? null },
      { pair: 'general-similarity', similarity: result.similarity_score }
    ].filter(entry => entry.similarity != null)

    const featureData = [
      { feature: 'TF-IDF Vectors', value: result.tfidf_features ?? 0, unit: 'dimensions', color: '#3b82f6' },
      { feature: 'Vocabulary Size', value: result.vocabulary_size ?? 0, unit: 'terms', color: '#10b981' },
      { feature: 'Sparsity', value: ((result.sparsity ?? 0) * 100).toFixed(1), unit: '%', color: '#8b5cf6' }
    ]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Similarity Bar Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Semantic Similarity Analysis</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={similarityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="pair" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <YAxis domain={[-1, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="similarity" fill="#8884d8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Feature Metrics Cards */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Feature Extraction Metrics</h3>
            <div className="space-y-3">
              {featureData.map((item, index) => (
                <div key={index} className="glass p-4 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">{item.feature}</span>
                    <span className="text-lg font-bold" style={{ color: item.color }}>
                      {item.value} {item.unit}
                    </span>
                  </div>
                </div>
              ))}

              <div className="glass p-4 rounded-lg">
                <div className="text-sm text-gray-400 mb-1">Accuracy Score</div>
                <div className="text-lg font-bold text-green-400">
                  {((result.semantic_accuracy ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center p-6 glass rounded-xl">
          <div className="text-xl text-white mb-2">{result.embedding_model || 'Embedding model not reported'}</div>
          <div className="text-sm text-gray-400 mb-4">
            {(result.tfidf_features ?? 0)}D TF-IDF features with {(result.vocabulary_size ?? 0)} vocabulary terms
          </div>
          <div className="text-sm text-gray-400">
            Method: {result.method || 'Not reported'}
          </div>
        </div>
      </div>
    )
  }

  // Default fallback
  return (
    <div className="text-center p-8">
      <h3 className="text-xl text-white mb-4">Algorithm Results</h3>
      <div className="glass p-6 rounded-xl">
        <div className="text-sm text-gray-400 mb-4">
          Status: {result.success ? '✅ Success' : '❌ Failed'}
        </div>
        <pre className="text-xs text-left bg-black/20 p-4 rounded overflow-auto max-h-40">
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    </div>
  )
}

export default ChartVisualization
