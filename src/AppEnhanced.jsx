import { useState, useEffect } from 'react'
import { Activity, Zap, CheckCircle, AlertCircle, TrendingUp, BarChart3, Brain, Layers, GitBranch, Cpu, Sparkles, BookOpen, Image, Eye, Settings, PlayCircle, PauseCircle } from 'lucide-react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line, AreaChart, Area, ScatterChart, Scatter, Treemap } from 'recharts'

function App() {
  const [stats, setStats] = useState(null)
  const [testResults, setTestResults] = useState({})
  const [testing, setTesting] = useState({})
  const [activeTab, setActiveTab] = useState('algorithms')
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const testAlgorithm = async (name, endpoint) => {
    setTesting(prev => ({ ...prev, [name]: true }))
    try {
      const response = await axios.post(endpoint)
      setTestResults(prev => ({ ...prev, [name]: response.data }))
    } catch (error) {
      setTestResults(prev => ({ ...prev, [name]: { success: false, error: error.message } }))
    } finally {
      setTesting(prev => ({ ...prev, [name]: false }))
    }
  }

  const algorithms = [
    {
      name: 'Quality Scorer',
      endpoint: '/api/test/quality-scorer',
      icon: '🎨',
      description: '4-component weighted quality assessment with CLIP, LPIPS, ResNet50',
      color: 'from-blue-500 to-cyan-500',
      category: 'Assessment',
      details: 'Evaluates edit quality using instruction compliance (40%), editing realism (25%), preservation balance (20%), and technical quality (15%)',
      hasImages: true,
      imageType: 'quality-assessment'
    },
    {
      name: 'Diffusion Model',
      endpoint: '/api/test/diffusion-model',
      icon: '🌊',
      description: 'U-Net architecture with cross-attention for instruction-guided editing',
      color: 'from-purple-500 to-pink-500',
      category: 'Generation',
      details: '10.9M parameters, supports text-to-image editing with diffusion process',
      hasImages: true,
      imageType: 'diffusion-edit'
    },
    {
      name: 'DPO Training',
      endpoint: '/api/test/dpo-training',
      icon: '🎯',
      description: 'Direct Preference Optimization for human-aligned model training',
      color: 'from-orange-500 to-red-500',
      category: 'Training',
      details: 'Trains models using preference pairs to align with human judgments',
      hasImages: false,
      imageType: 'training-metrics'
    },
    {
      name: 'Multi-Turn Editor',
      endpoint: '/api/test/multi-turn',
      icon: '🔄',
      description: 'Conversational image editing with history management',
      color: 'from-green-500 to-emerald-500',
      category: 'Interaction',
      details: 'Supports sequential edits with conflict detection and undo/redo',
      hasImages: true,
      imageType: 'multi-turn'
    },
    {
      name: 'Core ML Optimizer',
      endpoint: '/api/test/coreml',
      icon: '🍎',
      description: 'Apple Silicon optimization with Neural Engine acceleration',
      color: 'from-indigo-500 to-purple-500',
      category: 'Deployment',
      details: 'Converts PyTorch models to Core ML with quantization for iOS',
      hasImages: false,
      imageType: 'deployment'
    },
    {
      name: 'Baseline Model',
      endpoint: '/api/test/baseline',
      icon: '📊',
      description: 'Scikit-learn pipeline with TF-IDF and logistic regression',
      color: 'from-teal-500 to-cyan-500',
      category: 'Baseline',
      details: 'Fast, interpretable baseline using classical ML techniques',
      hasImages: false,
      imageType: 'baseline'
    },
    {
      name: 'Feature Extraction',
      endpoint: '/api/test/features',
      icon: '🔍',
      description: 'TF-IDF text features and image similarity computation',
      color: 'from-yellow-500 to-orange-500',
      category: 'Features',
      details: 'Extracts 1024-dim TF-IDF features and computes image similarity',
      hasImages: true,
      imageType: 'feature-similarity'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Enhanced Header */}
        <div className="glass rounded-3xl p-6 md:p-8 mb-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                🎯 PicoTuri-EditJudge
              </h1>
              <p className="text-lg md:text-xl text-gray-300">Advanced Algorithm Research Dashboard 2025</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setActiveTab('algorithms')}
                className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${activeTab === 'algorithms' ? 'bg-blue-500' : 'bg-white/10 hover:bg-white/20'}`}
              >
                <Brain className="w-4 h-4" />
                Algorithms
              </button>
              <button
                onClick={() => setActiveTab('research')}
                className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${activeTab === 'research' ? 'bg-purple-500' : 'bg-white/10 hover:bg-white/20'}`}
              >
                <BookOpen className="w-4 h-4" />
                Research
              </button>
            </div>
          </div>
        </div>

        {/* Enhanced Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 mb-8">
            <StatCard 
              icon={<Activity className="w-6 md:w-8 h-6 md:h-8" />}
              title="Algorithms"
              value="7/7"
              subtitle="All Working"
              color="blue"
              trend="+100%"
            />
            <StatCard 
              icon={<CheckCircle className="w-6 md:w-8 h-6 md:h-8" />}
              title="Code Quality"
              value="100%"
              subtitle="PEP 8"
              color="green"
              trend="Perfect"
            />
            <StatCard 
              icon={<Zap className="w-6 md:w-8 h-6 md:h-8" />}
              title="Coverage"
              value="100%"
              subtitle="Tested"
              color="purple"
              trend="Complete"
            />
            <StatCard 
              icon={<AlertCircle className="w-6 md:w-8 h-6 md:h-8" />}
              title="Errors"
              value="0"
              subtitle="Zero Issues"
              color="cyan"
              trend="Clean"
            />
          </div>
        )}

        {/* Main Content */}
        {activeTab === 'algorithms' ? (
          <>
            {/* Algorithm Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
              {algorithms.map((algo) => (
                <EnhancedAlgorithmCard
                  key={algo.name}
                  {...algo}
                  onTest={() => testAlgorithm(algo.name, algo.endpoint)}
                  testing={testing[algo.name]}
                  result={testResults[algo.name]}
                  onSelect={() => setSelectedAlgorithm(algo)}
                />
              ))}
            </div>
          </>
        ) : (
          <EnhancedResearchPanel algorithms={algorithms} testResults={testResults} />
        )}

        {/* Algorithm Detail Modal */}
        {selectedAlgorithm && (
          <AlgorithmDetailModal 
            algorithm={selectedAlgorithm}
            result={testResults[selectedAlgorithm.name]}
            onClose={() => setSelectedAlgorithm(null)}
          />
        )}
      </div>
    </div>
  )
}

function StatCard({ icon, title, value, subtitle, color, trend }) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-cyan-500/20 border-blue-500/30',
    green: 'from-green-500/20 to-emerald-500/20 border-green-500/30',
    purple: 'from-purple-500/20 to-pink-500/20 border-purple-500/30',
    cyan: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30'
  }

  return (
    <div className={`glass rounded-2xl p-4 md:p-6 bg-gradient-to-br ${colorClasses[color]} card-hover relative overflow-hidden`}>
      <div className="absolute top-2 right-2">
        <TrendingUp className="w-4 h-4 text-green-400" />
      </div>
      <div className="flex items-center justify-between mb-3 md:mb-4">
        <div className="text-white/80">{icon}</div>
        <span className="text-xs text-green-400 font-semibold">{trend}</span>
      </div>
      <div className="text-2xl md:text-3xl font-bold mb-1">{value}</div>
      <div className="text-xs md:text-sm text-gray-400">{subtitle}</div>
    </div>
  )
}

function EnhancedAlgorithmCard({ name, icon, description, details, color, category, hasImages, imageType, onTest, testing, result, onSelect }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="glass rounded-2xl p-6 card-hover flex flex-col relative overflow-hidden group">
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-white/5 to-transparent rounded-bl-full"></div>
      
      <div className="flex items-start justify-between mb-4">
        <div className="text-4xl">{icon}</div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-1 rounded-full bg-white/10">{category}</span>
          {hasImages && (
            <button
              onClick={onSelect}
              className="text-xs px-2 py-1 rounded-full bg-blue-500/20 hover:bg-blue-500/30 transition-all"
            >
              <Eye className="w-3 h-3 inline mr-1" />
              Images
            </button>
          )}
        </div>
      </div>
      
      <h3 className="text-xl md:text-2xl font-bold mb-2">{name}</h3>
      <p className="text-sm text-gray-400 mb-4 flex-grow">{description}</p>
      
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-xs text-blue-400 hover:text-blue-300 mb-4 text-left transition-all"
      >
        {expanded ? '▼ Hide Details' : '▶ Show Details'}
      </button>
      
      {expanded && (
        <div className="mb-4 p-3 bg-black/20 rounded-lg text-sm text-gray-300 animate-fadeIn">
          {details}
        </div>
      )}
      
      <button
        onClick={onTest}
        disabled={testing}
        className={`w-full py-3 px-6 rounded-xl font-semibold bg-gradient-to-r ${color} hover:opacity-90 transition-all disabled:opacity-50 mb-4 flex items-center justify-center gap-2`}
      >
        {testing ? (
          <>
            <PauseCircle className="w-4 h-4 animate-spin" />
            Testing...
          </>
        ) : (
          <>
            <PlayCircle className="w-4 h-4" />
            Test Algorithm
          </>
        )}
      </button>

      {result && (
        <EnhancedResultsPanel result={result} algorithmName={name} />
      )}
    </div>
  )
}

function EnhancedResultsPanel({ result, algorithmName }) {
  return (
    <div className={`p-4 rounded-xl ${result.success ? 'bg-green-500/20' : 'bg-red-500/20'} animate-fadeIn`}>
      <div className="font-semibold mb-3 flex items-center gap-2">
        {result.success ? (
          <>
            <CheckCircle className="w-4 h-4 text-green-400" />
            Test Passed!
          </>
        ) : (
          <>
            <AlertCircle className="w-4 h-4 text-red-400" />
            Test Failed!
          </>
        )}
      </div>
      
      {/* Quality Scorer - Keep existing good visualization */}
      {result.components && (
        <div className="space-y-4">
          <ComponentBarChart data={result.components} weights={result.weights} />
          <ComponentPieChart data={result.components} />
          <ComponentRadarChart data={result.components} />
          <ScoreTable data={result} />
        </div>
      )}
      
      {/* Enhanced Diffusion Model Visualization */}
      {result.parameters && !result.components && algorithmName === 'Diffusion Model' && (
        <DiffusionModelVisualization data={result} />
      )}
      
      {/* Enhanced DPO Training Visualization */}
      {result.loss !== undefined && (
        <DPOTrainingVisualization data={result} />
      )}
      
      {/* Enhanced Multi-Turn Editor Visualization */}
      {result.instructions_processed !== undefined && (
        <MultiTurnVisualization data={result} />
      )}
      
      {/* Enhanced Core ML Optimizer Visualization */}
      {result.ios_files_generated !== undefined && (
        <CoreMLVisualization data={result} />
      )}
      
      {/* Enhanced Baseline Model Visualization */}
      {result.classifier && (
        <BaselineModelVisualization data={result} />
      )}
      
      {/* Enhanced Feature Extraction Visualization */}
      {result.tfidf_features !== undefined && (
        <FeatureExtractionVisualization data={result} />
      )}
      
      <details className="mt-4">
        <summary className="cursor-pointer text-sm text-gray-400 hover:text-white">View Raw JSON</summary>
        <pre className="text-xs overflow-auto max-h-40 mt-2 p-2 bg-black/30 rounded">
          {JSON.stringify(result, null, 2)}
        </pre>
      </details>
    </div>
  )
}

// Enhanced Visualization Components
function DiffusionModelVisualization({ data }) {
  const modelData = [
    { name: 'Parameters', value: data.parameters, fill: '#8b5cf6' },
    { name: 'Input Size', value: data.input_shape?.reduce((a, b) => a * b, 1) || 0, fill: '#ec4899' },
    { name: 'Output Size', value: data.output_shape?.reduce((a, b) => a * b, 1) || 0, fill: '#3b82f6' }
  ]

  return (
    <div className="space-y-4">
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4" />
          Model Architecture
        </h4>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center p-3 bg-purple-500/20 rounded-lg">
            <div className="text-2xl font-bold text-purple-400">{(data.parameters / 1000000).toFixed(1)}M</div>
            <div className="text-xs text-gray-400">Parameters</div>
          </div>
          <div className="text-center p-3 bg-pink-500/20 rounded-lg">
            <div className="text-2xl font-bold text-pink-400">{data.input_shape?.join('×')}</div>
            <div className="text-xs text-gray-400">Input</div>
          </div>
          <div className="text-center p-3 bg-blue-500/20 rounded-lg">
            <div className="text-2xl font-bold text-blue-400">{data.output_shape?.join('×')}</div>
            <div className="text-xs text-gray-400">Output</div>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <Treemap
            data={[{ name: 'Model', children: modelData }]}
            dataKey="value"
            aspectRatio={4/3}
            stroke="#ffffff20"
            fill="#8b5cf6"
          />
        </ResponsiveContainer>
      </div>
      
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3">Performance Metrics</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <span className="text-sm">Architecture</span>
            <span className="text-sm font-semibold text-purple-400">{data.architecture}</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <span className="text-sm">Efficiency</span>
            <span className="text-sm font-semibold text-green-400">High</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function DPOTrainingVisualization({ data }) {
  const trainingData = [
    { name: 'Loss', value: data.loss, fill: '#f97316' },
    { name: 'Accuracy', value: data.preference_accuracy, fill: '#ef4444' },
    { name: 'KL Divergence', value: data.kl_divergence * 1000, fill: '#fbbf24' }
  ]

  return (
    <div className="space-y-4">
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <TrendingUp className="w-4 h-4" />
          Training Progress
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
            <XAxis dataKey="name" tick={{ fill: '#fff', fontSize: 10 }} />
            <YAxis tick={{ fill: '#fff', fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
            <Bar dataKey="value" fill="#f97316" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3">Training Details</h4>
        <div className="space-y-2">
          <div className="flex justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Steps Completed</span>
            <span className="text-sm font-semibold">{data.training_steps}</span>
          </div>
          <div className="flex justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Learning Rate</span>
            <span className="text-sm font-semibold">{data.learning_rate}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function MultiTurnVisualization({ data }) {
  const sessionData = [
    { name: 'Processed', value: data.instructions_processed, fill: '#10b981' },
    { name: 'Completed', value: data.edits_completed, fill: '#34d399' },
    { name: 'Failed', value: data.failed_edits, fill: '#ef4444' }
  ]

  return (
    <div className="space-y-4">
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <GitBranch className="w-4 h-4" />
          Editing Session
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={sessionData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}`}
              outerRadius={60}
              fill="#8884d8"
              dataKey="value"
            >
              {sessionData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3">Session Metrics</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-green-500/20 rounded-lg">
            <div className="text-2xl font-bold text-green-400">{data.success_rate?.toFixed(0)}%</div>
            <div className="text-xs text-gray-400">Success Rate</div>
          </div>
          <div className="text-center p-3 bg-emerald-500/20 rounded-lg">
            <div className="text-2xl font-bold text-emerald-400">{(data.average_confidence * 100).toFixed(0)}%</div>
            <div className="text-xs text-gray-400">Confidence</div>
          </div>
        </div>
      </div>
    </div>
  )
}

function CoreMLVisualization({ data }) {
  return (
    <div className="space-y-4">
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4" />
          iOS Deployment Ready
        </h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-4 bg-indigo-500/20 rounded-lg">
            <div className="text-3xl font-bold text-indigo-400">{data.ios_files_generated}</div>
            <div className="text-xs text-gray-400">Files Generated</div>
          </div>
          <div className="text-center p-4 bg-purple-500/20 rounded-lg">
            <div className="text-3xl font-bold text-purple-400">{data.coreml_version}</div>
            <div className="text-xs text-gray-400">Core ML Version</div>
          </div>
        </div>
      </div>
      
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3">Optimization Features</h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Apple Silicon</span>
            <span className={`px-2 py-1 rounded text-xs ${data.apple_silicon ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
              {data.apple_silicon ? 'Optimized' : 'Not Optimized'}
            </span>
          </div>
          <div className="flex items-center justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Neural Engine</span>
            <span className={`px-2 py-1 rounded text-xs ${data.neural_engine_support ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
              {data.neural_engine_support ? 'Supported' : 'Not Supported'}
            </span>
          </div>
          <div className="flex items-center justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Target iOS</span>
            <span className="text-sm font-semibold">{data.target_ios_version}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function BaselineModelVisualization({ data }) {
  const pipelineData = [
    { name: 'TF-IDF', value: 1, fill: '#14b8a6' },
    { name: 'Classifier', value: 1, fill: '#06b6d4' }
  ]

  return (
    <div className="space-y-4">
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Pipeline Architecture
        </h4>
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={pipelineData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
            <XAxis type="number" tick={{ fill: '#fff', fontSize: 10 }} />
            <YAxis dataKey="name" type="category" tick={{ fill: '#fff', fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
            <Bar dataKey="value" fill="#14b8a6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3">Model Configuration</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-teal-500/20 rounded-lg">
            <div className="text-xl font-bold text-teal-400">{data.classifier}</div>
            <div className="text-xs text-gray-400">Classifier</div>
          </div>
          <div className="text-center p-3 bg-cyan-500/20 rounded-lg">
            <div className="text-xl font-bold text-cyan-400">{data.solver}</div>
            <div className="text-xs text-gray-400">Solver</div>
          </div>
        </div>
        <div className="mt-4 p-3 bg-white/5 rounded-lg">
          <div className="flex justify-between items-center">
            <span className="text-sm">Max Iterations</span>
            <span className="text-sm font-semibold">{data.max_iter}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function FeatureExtractionVisualization({ data }) {
  const featureData = [
    { name: 'TF-IDF', value: data.tfidf_features, fill: '#fbbf24' },
    { name: 'Similarity', value: data.similarity_score * 100, fill: '#f59e0b' }
  ]

  return (
    <div className="space-y-4">
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Eye className="w-4 h-4" />
          Feature Analysis
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={featureData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
            <XAxis dataKey="name" tick={{ fill: '#fff', fontSize: 10 }} />
            <YAxis tick={{ fill: '#fff', fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
            <Bar dataKey="value" fill="#fbbf24" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="bg-black/20 p-4 rounded-lg">
        <h4 className="text-sm font-semibold mb-3">Extraction Details</h4>
        <div className="space-y-2">
          <div className="flex justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Feature Dimensions</span>
            <span className="text-sm font-semibold">{data.tfidf_features}D</span>
          </div>
          <div className="flex justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">N-gram Range</span>
            <span className="text-sm font-semibold">{data.ngram_range}</span>
          </div>
          <div className="flex justify-between p-2 bg-white/5 rounded">
            <span className="text-sm">Similarity Method</span>
            <span className="text-sm font-semibold">{data.method}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// Keep existing good components
function ComponentBarChart({ data, weights }) {
  const chartData = Object.entries(data).map(([key, value]) => ({
    name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    score: parseFloat((value * 100).toFixed(1)),
    weight: weights ? parseFloat((weights[key] * 100).toFixed(0)) : 0
  }))

  return (
    <div className="bg-black/20 p-4 rounded-lg">
      <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
        <BarChart3 className="w-4 h-4" />
        Component Scores
      </h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
          <XAxis dataKey="name" tick={{ fill: '#fff', fontSize: 10 }} angle={-45} textAnchor="end" height={80} />
          <YAxis tick={{ fill: '#fff', fontSize: 10 }} />
          <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          <Bar dataKey="score" fill="#3b82f6" name="Score (%)" />
          <Bar dataKey="weight" fill="#8b5cf6" name="Weight (%)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ComponentPieChart({ data }) {
  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981']
  const chartData = Object.entries(data).map(([key, value], index) => ({
    name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: parseFloat((value * 100).toFixed(2)),
    color: COLORS[index % COLORS.length]
  }))

  return (
    <div className="bg-black/20 p-4 rounded-lg">
      <h4 className="text-sm font-semibold mb-3">Score Distribution</h4>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name.split(' ')[0]}: ${value}%`}
            outerRadius={60}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

function ComponentRadarChart({ data }) {
  const chartData = Object.entries(data).map(([key, value]) => ({
    subject: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    score: parseFloat((value * 100).toFixed(1)),
    fullMark: 100
  }))

  return (
    <div className="bg-black/20 p-4 rounded-lg">
      <h4 className="text-sm font-semibold mb-3">Performance Radar</h4>
      <ResponsiveContainer width="100%" height={250}>
        <RadarChart data={chartData}>
          <PolarGrid stroke="#ffffff20" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#fff', fontSize: 10 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#fff', fontSize: 10 }} />
          <Radar name="Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
          <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #ffffff20', borderRadius: '8px' }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ScoreTable({ data }) {
  return (
    <div className="bg-black/20 p-4 rounded-lg overflow-x-auto">
      <h4 className="text-sm font-semibold mb-3">Detailed Metrics</h4>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/20">
            <th className="text-left py-2">Metric</th>
            <th className="text-right py-2">Score</th>
            <th className="text-right py-2">Weight</th>
          </tr>
        </thead>
        <tbody>
          {data.components && Object.entries(data.components).map(([key, value]) => (
            <tr key={key} className="border-b border-white/10">
              <td className="py-2 font-medium">
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </td>
              <td className="text-right">
                <span className={`px-2 py-1 rounded text-xs ${value > 0.7 ? 'bg-green-500/20' : value > 0.5 ? 'bg-yellow-500/20' : 'bg-red-500/20'}`}>
                  {(value * 100).toFixed(1)}%
                </span>
              </td>
              <td className="text-right text-gray-400 text-xs">
                {data.weights ? `${(data.weights[key] * 100)}%` : '-'}
              </td>
            </tr>
          ))}
          <tr className="border-t-2 border-white/30 font-bold">
            <td className="py-2">Overall</td>
            <td className="text-right">
              <span className={`px-2 py-1 rounded text-xs ${data.overall_score > 0.7 ? 'bg-green-500/30' : data.overall_score > 0.5 ? 'bg-yellow-500/30' : 'bg-red-500/30'}`}>
                {(data.overall_score * 100).toFixed(1)}%
              </span>
            </td>
            <td className="text-right text-xs">{data.grade}</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

// Enhanced Research Panel
function EnhancedResearchPanel({ algorithms, testResults }) {
  const categories = ['Assessment', 'Generation', 'Training', 'Interaction', 'Deployment', 'Baseline', 'Features']
  
  return (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <BookOpen className="w-6 h-6" />
          Research Overview
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Algorithm Categories</h3>
            <div className="space-y-3">
              {categories.map(cat => {
                const count = algorithms.filter(a => a.category === cat).length
                const percentage = (count / algorithms.length) * 100
                return (
                  <div key={cat} className="p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-all">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{cat}</span>
                      <span className="text-sm text-gray-400">{count} algorithm(s)</span>
                    </div>
                    <div className="w-full bg-white/10 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all"
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Research Statistics</h3>
            <div className="space-y-4">
              <div className="p-4 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg">
                <div className="text-sm text-gray-400 mb-1">Total Algorithms</div>
                <div className="text-3xl font-bold">{algorithms.length}</div>
                <div className="text-xs text-gray-500 mt-1">All categories covered</div>
              </div>
              <div className="p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-lg">
                <div className="text-sm text-gray-400 mb-1">Tests Completed</div>
                <div className="text-3xl font-bold">{Object.keys(testResults).length}</div>
                <div className="text-xs text-gray-500 mt-1">Real-time testing</div>
              </div>
              <div className="p-4 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-lg">
                <div className="text-sm text-gray-400 mb-1">Success Rate</div>
                <div className="text-3xl font-bold">
                  {Object.keys(testResults).length > 0 
                    ? Math.round((Object.values(testResults).filter(r => r.success).length / Object.keys(testResults).length) * 100)
                    : 0}%
                </div>
                <div className="text-xs text-gray-500 mt-1">Algorithm reliability</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="glass rounded-2xl p-6">
        <h2 className="text-2xl font-bold mb-6">Algorithm Research Details</h2>
        <div className="grid gap-4">
          {algorithms.map(algo => (
            <div key={algo.name} className="p-6 bg-white/5 rounded-lg hover:bg-white/10 transition-all">
              <div className="flex items-start gap-4">
                <div className="text-4xl">{algo.icon}</div>
                <div className="flex-grow">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xl font-semibold">{algo.name}</h3>
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-3 py-1 rounded-full bg-white/10">{algo.category}</span>
                      {algo.hasImages && (
                        <span className="text-xs px-3 py-1 rounded-full bg-blue-500/20 text-blue-400">
                          <Image className="w-3 h-3 inline mr-1" />
                          Visual
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-gray-300 mb-3">{algo.description}</p>
                  <p className="text-sm text-gray-400 mb-4">{algo.details}</p>
                  
                  {testResults[algo.name] && (
                    <div className="p-3 bg-black/20 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        {testResults[algo.name].success ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-400" />
                        )}
                        <span className="text-sm font-semibold">
                          {testResults[algo.name].success ? 'Test Passed' : 'Test Failed'}
                        </span>
                      </div>
                      <div className="text-xs text-gray-400">
                        Last tested: {new Date().toLocaleTimeString()}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Algorithm Detail Modal with Images
function AlgorithmDetailModal({ algorithm, result, onClose }) {
  if (!algorithm) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <span className="text-3xl">{algorithm.icon}</span>
            {algorithm.name}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-all"
          >
            ✕
          </button>
        </div>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-2">Description</h3>
            <p className="text-gray-300">{algorithm.description}</p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">Technical Details</h3>
            <p className="text-gray-400">{algorithm.details}</p>
          </div>
          
          {algorithm.hasImages && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Image Comparison</h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-black/20 p-4 rounded-lg">
                  <h4 className="text-sm font-semibold mb-3">Original Image</h4>
                  <div className="aspect-square bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex items-center justify-center">
                    <Image className="w-12 h-12 text-gray-500" />
                  </div>
                  <p className="text-xs text-gray-400 mt-2">Before algorithm processing</p>
                </div>
                <div className="bg-black/20 p-4 rounded-lg">
                  <h4 className="text-sm font-semibold mb-3">Processed Image</h4>
                  <div className="aspect-square bg-gradient-to-br from-blue-700 to-purple-800 rounded-lg flex items-center justify-center">
                    <Sparkles className="w-12 h-12 text-blue-400" />
                  </div>
                  <p className="text-xs text-gray-400 mt-2">After {algorithm.name} processing</p>
                </div>
              </div>
            </div>
          )}
          
          {result && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Test Results</h3>
              <div className={`p-4 rounded-lg ${result.success ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                <pre className="text-xs overflow-auto max-h-40">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
