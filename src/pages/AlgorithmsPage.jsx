import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckCircle, PauseCircle, PlayCircle } from 'lucide-react'
import api from '../utils/api'
import { trapTabKey } from '../utils/focusTrap'
import ChartVisualization from '../components/algorithms/ChartVisualization'
import { isValidAlgorithmResult } from '../components/algorithms/algorithmResult'

const supportsDialogModal = typeof HTMLDialogElement !== 'undefined'
  && typeof HTMLDialogElement.prototype.showModal === 'function'

const AlgorithmsPage = () => {
  const [testResults, setTestResults] = useState({})
  const [testing, setTesting] = useState({})
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null)

  const requests = useRef(new Map())
  const closeResults = useCallback(() => setSelectedAlgorithm(null), [])

  useEffect(() => {
    const activeRequests = requests.current
    return () => {
      activeRequests.forEach(controller => controller.abort())
      activeRequests.clear()
    }
  }, [])

  const testAlgorithm = async (name, endpoint) => {
    if (requests.current.has(name)) return
    const controller = new AbortController()
    requests.current.set(name, controller)
    setTesting(prev => ({ ...prev, [name]: true }))
    try {
      const response = await api.post(endpoint, undefined, { signal: controller.signal })
      if (controller.signal.aborted) return
      if (!isValidAlgorithmResult(name, response.data)) {
        setTestResults(prev => ({
          ...prev,
          [name]: { success: false, error: 'The API returned an unsuccessful or invalid result. Please try again.' }
        }))
        return
      }
      const isDemo = response.data.data_source === 'demo'
      setTestResults(prev => ({
        ...prev,
        [name]: { ...response.data, isRealData: !isDemo, isMockData: isDemo }
      }))
    } catch (error) {
      if (controller.signal.aborted) return
      // Keep the offline demo available, but never mask an API rejection as success.
      const result = error.response
        ? { success: false, error: `The API request failed (${error.response.status}). Please try again.` }
        : { ...getMockResponse(name), isRealData: false, isMockData: true, offline: true }
      setTestResults(prev => ({ ...prev, [name]: result }))
    } finally {
      requests.current.delete(name)
      if (!controller.signal.aborted) {
        setTesting(prev => ({ ...prev, [name]: false }))
      }
    }
  }

  const getMockResponse = (algorithmName) => {
    switch (algorithmName) {
      case 'Quality Scorer':
        return {
          success: true,
          overall_score: 0.91,
          components: {
            instruction_compliance: 0.92,
            editing_realism: 0.89,
            preservation_balance: 0.88,
            technical_quality: 0.9
          },
          grade: 'A',
          recommendation: 'Excellent balance across fidelity and realism'
        }
      case 'Diffusion Model':
        return {
          success: true,
          parameters: 10_900_000,
          architecture: 'U-Net with cross-attention',
          inference_time_ms: 83.2,
          quality_score: 4.6
        }
      case 'DPO Training':
        return {
          success: true,
          loss: 0.61,
          preference_accuracy: 0.68,
          training_steps: 12
        }
      case 'Multi-Turn Editor':
        return {
          success: true,
          instructions_processed: 3,
          edits_completed: 3,
          failed_edits: 0,
          success_rate: 93.4,
          average_confidence: 0.88,
          processing_time_ms: 148.3
        }
      case 'Core ML Optimizer':
        return {
          success: true,
          ios_files_generated: 3,
          apple_silicon: true,
          neural_engine_support: true,
          quantization_applied: true,
          coreml_version: '7.1',
          target_ios_version: '17.0+',
          model_size_reduction: 0.72,
          compression_ratio: 3.6
        }
      case 'Baseline Model':
        return {
          success: true,
          classifier: 'LogisticRegression',
          training_accuracy: 0.982,
          validation_accuracy: 0.957,
          roc_auc: 0.941,
          f1_score: 0.924,
          pipeline_steps: 3,
          vocabulary_size: 1850
        }
      case 'Feature Extraction':
        return {
          success: true,
          tfidf_features: 1024,
          similarity_score: 0.87,
          semantic_accuracy: 0.67,
          vocabulary_size: 1850,
          sparsity: 0.74,
          between_group_similarity: 0.23,
          within_group_similarity_brighten: 0.91,
          within_group_similarity_darken: 0.89
        }
      default:
        return { success: false, error: 'Mock data not available' }
    }
  }

  const algorithms = [
    {
      name: 'Quality Scorer',
      endpoint: '/api/test/quality-scorer',
      icon: '🎨',
      description: '4-component weighted quality assessment',
      color: 'from-blue-500 to-cyan-500',
      category: 'Assessment'
    },
    {
      name: 'Diffusion Model',
      endpoint: '/api/test/diffusion-model',
      icon: '🌊',
      description: 'U-Net architecture with cross-attention',
      color: 'from-purple-500 to-pink-500',
      category: 'Generation'
    },
    {
      name: 'DPO Training',
      endpoint: '/api/test/dpo-training',
      icon: '🎯',
      description: 'Direct Preference Optimization training',
      color: 'from-orange-500 to-red-500',
      category: 'Training'
    },
    {
      name: 'Multi-Turn Editor',
      endpoint: '/api/test/multi-turn',
      icon: '🔄',
      description: 'Conversational image editing',
      color: 'from-green-500 to-emerald-500',
      category: 'Interaction'
    },
    {
      name: 'Core ML Optimizer',
      endpoint: '/api/test/coreml',
      icon: '🍎',
      description: 'Apple Silicon optimization',
      color: 'from-indigo-500 to-purple-500',
      category: 'Deployment'
    },
    {
      name: 'Baseline Model',
      endpoint: '/api/test/baseline',
      icon: '📊',
      description: 'Scikit-learn pipeline baseline',
      color: 'from-teal-500 to-cyan-500',
      category: 'Baseline'
    },
    {
      name: 'Feature Extraction',
      endpoint: '/api/test/features',
      icon: '🔍',
      description: 'TF-IDF text features and similarity',
      color: 'from-yellow-500 to-orange-500',
      category: 'Features'
    }
  ]

  return (
    <div className="lg:ml-80 min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="text-center">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
              🧪 Algorithm Testing Suite
            </h1>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Test and analyze all 7 cutting-edge AI algorithms for image editing quality assessment
            </p>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12 max-w-6xl mx-auto">
          <StatCard number="7" label="Algorithms" color="blue" />
          <StatCard number={Object.values(testResults).filter(result => result.success && !result.isMockData).length} label="Completed" color="green" />
          <StatCard number={Object.values(testResults).filter(result => result.isMockData).length} label="Demo Results" color="purple" />
          <StatCard number={Object.values(testResults).filter(result => !result.success).length} label="Failed" color="orange" />
        </div>

        {/* Algorithm Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6">
          {algorithms.map((algo, index) => (
            <AlgorithmCard
              key={algo.name}
              {...algo}
              onTest={() => testAlgorithm(algo.name, algo.endpoint)}
              testing={testing[algo.name]}
              result={testResults[algo.name]}
              onSelect={() => setSelectedAlgorithm(algo)}
              index={index}
            />
          ))}
        </div>

        {/* Results Modal */}
        {selectedAlgorithm && testResults[selectedAlgorithm.name] && (
          <AlgorithmResultsModal
            algorithm={selectedAlgorithm}
            result={testResults[selectedAlgorithm.name]}
            onClose={closeResults}
          />
        )}
      </div>
    </div>
  )
}

const statColors = {
  blue: 'text-blue-400',
  green: 'text-green-400',
  purple: 'text-purple-400',
  orange: 'text-orange-400'
}

const StatCard = ({ number, label, color }) => (
  <div className="glass rounded-xl p-6 text-center">
    <div className={`text-3xl font-bold ${statColors[color]} mb-2`}>{number}</div>
    <div className="text-sm text-gray-400">{label}</div>
  </div>
)

const AlgorithmCard = ({ name, icon, description, color, category, onTest, testing, result, onSelect, index }) => (
  <div className={`glass rounded-xl p-6 hover:scale-105 transition-all duration-300 ${
    result?.isMockData ? 'ring-2 ring-blue-400/50' : result?.success ? 'ring-2 ring-green-400/50' : ''
  } ${testing ? 'animate-pulse' : ''}`}
       style={{ animationDelay: `${index * 100}ms` }}
  >
    <div className="flex flex-col items-center text-center">
      <div className={`text-4xl ${result?.success && !result.isMockData ? 'animate-bounce' : ''}`} style={{ animationDelay: `${index * 50}ms` }}>
        {icon}
      </div>

      <h3 className="text-lg font-semibold text-white mt-4 mb-2">{name}</h3>
      <p className="text-sm text-gray-400 mb-4">{description}</p>

      <span className="px-3 py-1 bg-white/10 text-xs rounded-full text-gray-300 mb-4">
        {category}
      </span>

      <div className="flex flex-col gap-3 w-full">
        <button
          type="button"
          onClick={onTest}
          disabled={testing}
          className={`px-4 py-2 rounded-lg font-medium transition-all w-full ${
            testing
              ? 'bg-gray-500/50 cursor-not-allowed'
              : result?.success && !result.isMockData
              ? 'bg-green-500/50 hover:bg-green-500/70'
              : `bg-gradient-to-r ${color} hover:scale-105`
          }`}
        >
          {testing ? (
            <>
              <PauseCircle className="w-4 h-4 inline mr-2" />
              Testing...
            </>
          ) : result?.success && !result.isMockData ? (
            <>
              <CheckCircle className="w-4 h-4 inline mr-2" />
              Tested
            </>
          ) : (
            <>
              <PlayCircle className="w-4 h-4 inline mr-2" />
              {result ? 'Try Again' : 'Test Now'}
            </>
          )}
        </button>

        {result?.error && <p role="alert" className="text-sm text-red-300">{result.error}</p>}
        {result?.success && (
          <button
            type="button"
            onClick={onSelect}
            className="px-4 py-2 rounded-lg font-medium transition-all w-full bg-blue-500/50 hover:bg-blue-500/70"
          >
            View Results
          </button>
        )}
      </div>

      {/* Status indicator */}
      <div className="mt-4 flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${
          testing ? 'bg-yellow-400 animate-pulse' :
          result?.isMockData ? 'bg-blue-400' : result?.success ? 'bg-green-400' : result ? 'bg-red-400' : 'bg-gray-400'
        }`}></div>
        <span className="text-xs text-gray-400">
          {testing ? 'Running...' : result?.isMockData ? 'Demo data' : result?.success ? 'Complete' : result ? 'Failed' : 'Ready to test'}
        </span>
      </div>
    </div>
  </div>
)

const AlgorithmResultsModal = ({ algorithm, result, onClose }) => {
  const dialogRef = useRef(null)
  const closeRef = useRef(null)

  useEffect(() => {
    const dialog = dialogRef.current
    const previouslyFocused = document.activeElement
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    if (dialog && supportsDialogModal) dialog.showModal()
    closeRef.current?.focus()
    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      trapTabKey(event, dialog)
    }
    const onCancel = (event) => {
      event.preventDefault()
      onClose()
    }
    document.addEventListener('keydown', onKey, true)
    dialog?.addEventListener('cancel', onCancel)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      dialog?.removeEventListener('cancel', onCancel)
      document.body.style.overflow = previousOverflow
      if (dialog?.open && typeof dialog.close === 'function') dialog.close()
      if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus()
    }
  }, [onClose])

  return (
  <dialog
    ref={dialogRef}
    open={supportsDialogModal ? undefined : true}
    aria-label={`${algorithm.name} results`}
    className="fixed inset-0 z-50 m-0 flex h-full max-h-none w-full max-w-none items-center justify-center border-0 bg-black/80 p-4 text-white backdrop:bg-black/80"
  >
    <button
      type="button"
      tabIndex={-1}
      aria-label="Dismiss results"
      className="absolute inset-0 border-0 bg-transparent p-0"
      onClick={onClose}
    />
    <div className="glass relative z-10 max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-3xl p-8">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">
            {algorithm.icon} {algorithm.name}
          </h2>
          <p className="text-gray-400">{algorithm.description}</p>
        </div>
        <button
          ref={closeRef}
          type="button"
          onClick={onClose}
          aria-label="Close results"
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {result.isMockData && (
        <p role="status" className="mb-4 text-amber-300">
          {result.offline ? 'API unavailable. Showing demo data.' : 'Demo data supplied by the API.'} These values are illustrative, not measured algorithm results.
        </p>
      )}
      <ChartVisualization algorithm={algorithm} result={result} />
    </div>
  </dialog>
  )
}

export default AlgorithmsPage
