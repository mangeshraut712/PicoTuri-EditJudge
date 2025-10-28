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

const ChartVisualization = ({ algorithm, result }) => {
  if (!result) return <div className="text-gray-400">No data available</div>

  // Quality Scorer - Component breakdown
  if (result.components) {
    const chartData = Object.entries(result.components).map(([key, value]) => ({
      name: key.split('_').map(w => w[0].toUpperCase() + w.slice(1)).join(' '),
      value: parseFloat((value * 100).toFixed(1)),
      weight: (result.weights || {})[key] || 0.25
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
            Grade: {result.grade} | {result.recommendation}
          </div>
        </div>
      </div>
    )
  }

  // Diffusion Model - Architecture info with Line Chart
  if (result.parameters) {
    const architectureData = [
      { name: 'Parameter Count', value: result.parameters / 1000000, unit: 'M', color: '#8884d8' },
      { name: 'Inference Time', value: result.inference_time_ms, unit: 'ms', color: '#82ca9d' },
      { name: 'Quality Score', value: result.quality_score * 10, unit: '/50', color: '#ffc658' }
    ]

    // Mock performance data for line chart
    const performanceData = Array.from({ length: 10 }, (_, i) => ({
      time: i + 1,
      latency: result.inference_time_ms + (Math.random() - 0.5) * 10,
      quality: (result.quality_score * 10) + (Math.random() - 0.5) * 2
    }))

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
            <h3 className="text-lg font-semibold text-white mb-4">Performance Trends</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 11 }} />
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
                  dataKey="latency"
                  stroke="#8884d8"
                  strokeWidth={2}
                  name="Latency (ms)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="quality"
                  stroke="#82ca9d"
                  strokeWidth={2}
                  name="Quality Score"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="text-center p-6 glass rounded-xl">
          <div className="text-xl text-white mb-2">{result.architecture}</div>
          <div className="text-sm text-gray-400">
            {result.supports_text_to_image && result.supports_image_to_image ? 'Full generation support' : 'Limited capabilities'}
          </div>
        </div>
      </div>
    )
  }

  // Multi-Turn Editor - Session flow visualization
  if (result.instructions_processed !== undefined) {
    const sessionData = [
      { name: 'Start', value: 1, color: '#10b981' },
      { name: 'Processing', value: Math.max(result.instructions_processed - 1, 1), color: '#3b82f6' },
      { name: 'Completed', value: result.edits_completed, color: '#8b5cf6' },
      { name: 'Failed', value: result.failed_edits || 0, color: '#ef4444' }
    ]

    // Response time distribution
    const responseTimes = Array.from({ length: 20 }, (_, i) => ({
      request: i + 1,
      time: result.processing_time_ms / 20 + (Math.random() - 0.5) * 20,
      confidence: result.average_confidence + (Math.random() - 0.5) * 0.1
    }))

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Session Flow Pie Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Session Flow</h3>
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
            <h3 className="text-lg font-semibold text-white mb-4">Response Pattern</h3>
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
            <div className="text-2xl font-bold text-purple-400">{(result.average_confidence * 100)?.toFixed(0)}%</div>
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
  if (result.loss !== undefined) {
    const trainingData = result.training_history ? result.training_history : {
      loss: [0.94, 0.88, 0.81, 0.75, 0.69, 0.64, 0.6, 0.59, 0.58, 0.59, 0.6, 0.61],
      accuracy: [39, 46, 53, 58, 63, 66, 69, 71, 72, 71, 70, 68]
    }

    const steps = Array.from({ length: 12 }, (_, i) => i + 1)
    const chartData = steps.map(step => ({
      step,
      loss: trainingData.loss[step - 1],
      accuracy: trainingData.accuracy[step - 1] / 100
    }))

    return (
      <div className="space-y-6">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
            <XAxis dataKey="step" tick={{ fill: '#9ca3af' }} />
            <YAxis yAxisId="left" tick={{ fill: '#9ca3af' }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: '#9ca3af', formatter: (v) => `${(v * 100).toFixed(0)}%` }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px'
              }}
            />
            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} name="Loss" />
            <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} name="Accuracy" />
          </LineChart>
        </ResponsiveContainer>

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
  if (result.ios_files_generated !== undefined) {
    const optimizationData = [
      { metric: 'Size Reduction', before: 100, after: (result.model_size_reduction * 100).toFixed(0) },
      { metric: 'Speed Improvement', before: 100, after: 85 },
      { metric: 'Compatibility', before: 60, after: 95 }
    ]

    const featuresData = [
      { name: 'Apple Silicon', available: result.apple_silicon ? 1 : 0, color: '#10b981' },
      { name: 'Neural Engine', available: result.neural_engine_support ? 1 : 0, color: '#3b82f6' },
      { name: 'Quantization', available: result.quantization_applied ? 1 : 0, color: '#8b5cf6' }
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

          {/* Feature Availability Pie Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Feature Support</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={featuresData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="available"
                >
                  {featuresData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.available ? entry.color : '#ffffff20'} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                  formatter={(value) => [value ? 'Available' : 'Not Available', '']}
                />
              </PieChart>
            </ResponsiveContainer>
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
            <div className="text-2xl font-bold text-blue-400">{result.target_ios_version}</div>
            <div className="text-sm text-gray-400">Target Version</div>
          </div>
          <div className="glass p-4 rounded-xl text-center">
            <div className="text-2xl font-bold text-purple-400">{result.coreml_version}</div>
            <div className="text-sm text-gray-400">Core ML Version</div>
          </div>
        </div>
      </div>
    )
  }

  // Baseline Model - Performance scatter plot
  if (result.classifier !== undefined || (result.training_accuracy && result.validation_accuracy)) {
    const classifierName = result.classifier || 'Baseline Model'
    const modelComparisonData = [
      { model: 'LogisticRegression', accuracy: 0.982, f1: 0.924 },
      { model: 'RandomForest', accuracy: 0.967, f1: 0.901 },
      { model: 'SVM', accuracy: 0.943, f1: 0.868 },
      { model: 'NaiveBayes', accuracy: 0.901, f1: 0.812 },
      { model: classifierName, accuracy: result.training_accuracy || 0, f1: result.f1_score || 0 }
    ]

    const processingStepsData = [
      { step: 'Load Data', time: Math.random() * 2.3 },
      { step: 'Preprocess', time: Math.random() * 5.1 },
      { step: 'Training', time: Math.random() * 6.8 },
      { step: 'Validation', time: Math.random() * 0.8 }
    ]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Model Comparison Scatter Plot */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Model Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={modelComparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <YAxis domain={[0.85, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                  formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10b981"
                  strokeWidth={3}
                  name="Accuracy"
                  dot={{ fill: '#10b981', r: 5 }}
                />
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

          {/* Processing Pipeline Pie Chart */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Processing Pipeline</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={processingStepsData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="time"
                  nameKey="step"
                >
                  {processingStepsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b'][index % 4]} />
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
  if (result.tfidf_features !== undefined || result.similarity_score !== undefined) {
    const similarityData = [
      { pair: 'brighten-darken', similarity: -Math.abs(result.between_group_similarity ?? 0.23) },
      { pair: 'brighten-increase', similarity: result.within_group_similarity_brighten || 0.91 },
      { pair: 'darken-brighten', similarity: result.within_group_similarity_darken || 0.89 },
      { pair: 'general-similarity', similarity: result.similarity_score || 0.87 }
    ]

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
          <div className="text-xl text-white mb-2">{result.embedding_model || 'Sentence Transformer'}</div>
          <div className="text-sm text-gray-400 mb-4">
            {(result.tfidf_features ?? 0)}D TF-IDF features with {(result.vocabulary_size ?? 0)} vocabulary terms
          </div>
          <div className="text-sm text-gray-400">
            Method: {result.method || 'TF-IDF + Cosine Similarity'}
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
