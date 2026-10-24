import { render, screen } from '@testing-library/react'
import { Line, Tooltip, YAxis } from 'recharts'
import ChartVisualization from '../components/algorithms/ChartVisualization'
import { isValidAlgorithmResult } from '../components/algorithms/algorithmResult'
import { algorithmResults } from './algorithmFixtures'

// Inspect the chart's data and scale contract without depending on SVG layout in jsdom.
jest.mock('recharts', () => {
  const React = require('react')
  const chart = (name) => {
    function MockChart({ children, data }) {
      return React.createElement('div', {
        'data-testid': name,
        'data-points': data ? JSON.stringify(data) : undefined
      }, children)
    }
    MockChart.displayName = name
    return MockChart
  }
  function EmptyChart() {
    return null
  }
  return {
    ResponsiveContainer: chart('container'),
    LineChart: chart('line-chart'),
    BarChart: chart('bar-chart'),
    RadarChart: chart('radar-chart'),
    PieChart: chart('pie-chart'),
    Pie: chart('pie'),
    Line: jest.fn(() => null),
    YAxis: jest.fn(() => null),
    Tooltip: jest.fn(() => null),
    XAxis: EmptyChart,
    CartesianGrid: EmptyChart,
    PolarGrid: EmptyChart,
    PolarAngleAxis: EmptyChart,
    PolarRadiusAxis: EmptyChart,
    Radar: EmptyChart,
    Bar: EmptyChart,
    Cell: EmptyChart
  }
})

const show = (name, overrides = {}) => render(
  <ChartVisualization algorithm={{ name }} result={{ ...algorithmResults[name], ...overrides }} />
)
const points = type => JSON.parse(screen.getByTestId(type).dataset.points)

beforeEach(() => jest.clearAllMocks())

test.each(Object.keys(algorithmResults))('renders the actual %s API schema without invalid values', name => {
  const { container } = show(name)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(container).not.toHaveTextContent(/NaN|Infinity|undefined/)
})

test('distinguishes absent results from invalid or unsupported results', () => {
  const { rerender } = render(<ChartVisualization algorithm={{ name: 'Quality Scorer' }} result={null} />)
  expect(screen.getByText('No data available')).toBeInTheDocument()
  rerender(<ChartVisualization algorithm={{ name: 'Unknown' }} result={{ success: true }} />)
  expect(screen.getByRole('alert')).toHaveTextContent('No valid algorithm results')
  rerender(<ChartVisualization result={algorithmResults['Quality Scorer']} />)
  expect(screen.getByRole('alert')).toBeInTheDocument()
})

test.each([
  ['Quality Scorer', { overall_score: '0.91' }],
  ['Quality Scorer', { overall_score: null }],
  ['Quality Scorer', { overall_score: Infinity }],
  ['Quality Scorer', { components: {} }],
  ['Quality Scorer', { components: { score: 2 } }],
  ['Quality Scorer', { recommendation: {} }],
  ['Diffusion Model', { parameters: -1 }],
  ['Diffusion Model', { inference_time_ms: NaN }],
  ['Diffusion Model', { quality_score: 6 }],
  ['Diffusion Model', { supports_text_to_image: 'true' }],
  ['Multi-Turn Editor', { failed_edits: 1.5 }],
  ['Multi-Turn Editor', { average_confidence: 2 }],
  ['Multi-Turn Editor', { success_rate: 101 }],
  ['Core ML Optimizer', { model_size_reduction: 1.2 }],
  ['Core ML Optimizer', { apple_silicon: 'yes' }],
  ['Feature Extraction', { similarity_score: -2 }],
  ['Feature Extraction', { within_group_similarity_darken: '0.89' }],
  ['DPO Training', { training_history: { loss: [], accuracy: [] } }],
  ['DPO Training', { training_history: { loss: [0.6], accuracy: [] } }],
  ['DPO Training', { training_history: { loss: [-1], accuracy: [50] } }],
  ['DPO Training', { training_history: { loss: [0.6], accuracy: [101] } }]
])('rejects malformed %s metrics: %j', (name, overrides) => {
  show(name, overrides)
  expect(screen.getByRole('alert')).toHaveTextContent('No valid algorithm results')
  expect(screen.queryByTestId('container')).not.toBeInTheDocument()
})

test.each([null, [], '<html>Not JSON</html>', { success: false }])('rejects invalid result objects: %j', result => {
  expect(isValidAlgorithmResult('Quality Scorer', result)).toBe(false)
})

test('preserves zero scores and converts component fractions to percentages', () => {
  show('Quality Scorer', { overall_score: 0, components: { editing_realism: 0, technical_quality: 0.5 } })
  expect(points('bar-chart')).toEqual([
    { name: 'Editing Realism', value: 0 },
    { name: 'Technical Quality', value: 50 }
  ])
  expect(screen.getByText('0.0%')).toBeInTheDocument()
})

test('plots only the reported diffusion sample with separate latency and quality scales', () => {
  show('Diffusion Model')
  expect(points('line-chart')).toEqual([{ sample: 'Reported result', latency: 83.2, quality: 4.6 }])
  expect(screen.getByText('4.6 /5')).toBeInTheDocument()
  expect(Line.mock.calls.map(([props]) => [props.dataKey, props.yAxisId])).toEqual([
    ['latency', 'latency'], ['quality', 'quality']
  ])
  expect(YAxis.mock.calls.map(([props]) => props)).toEqual(expect.arrayContaining([
    expect.objectContaining({ yAxisId: 'quality', domain: [0, 5] }),
    expect.objectContaining({ yAxisId: 'latency', unit: ' ms' })
  ]))
})

test('keeps explicitly unsupported diffusion capabilities distinct from missing capabilities', () => {
  show('Diffusion Model', { supports_text_to_image: false })
  expect(screen.getByText(/Text-to-image: Not supported/)).toHaveTextContent('Image-to-image: Not reported')
})

test('plots the reported session total without inventing a response-time distribution or failed edits', () => {
  show('Multi-Turn Editor')
  expect(points('line-chart')).toEqual([{ request: 'Session total', time: 148.3 }])
  expect(points('pie')).toEqual([{ name: 'Completed', value: 3, color: '#8b5cf6' }])
  expect(screen.getByText('Not reported')).toBeInTheDocument()
  expect(screen.getByText('Failed edit count was not reported.')).toBeInTheDocument()
})

test('shows an empty session explicitly and preserves a reported zero confidence', () => {
  show('Multi-Turn Editor', { instructions_processed: 0, edits_completed: 0, failed_edits: 0, average_confidence: 0 })
  expect(screen.getByText('No completed or failed edits reported.')).toBeInTheDocument()
  expect(screen.getByText('0%')).toBeInTheDocument()
})

test('does not invent training history when the API only supplies a final result', () => {
  show('DPO Training')
  expect(points('line-chart')).toEqual([{ step: 12, loss: 0.61, accuracy: 0.68 }])
  expect(screen.getByText(/per-step training history is unavailable/)).toBeInTheDocument()
})

test('uses the supplied history length and formats percentage history and final accuracy consistently', () => {
  show('DPO Training', { training_history: { loss: [0.9, 0.7, 0.61], accuracy: [40, 60, 68] } })
  expect(points('line-chart')).toEqual([
    { step: 1, loss: 0.9, accuracy: 0.4 },
    { step: 2, loss: 0.7, accuracy: 0.6 },
    { step: 3, loss: 0.61, accuracy: 0.68 }
  ])
  const accuracyAxis = YAxis.mock.calls.find(([props]) => props.yAxisId === 'right')[0]
  expect(accuracyAxis.tickFormatter(0.68)).toBe('68%')
  const tooltip = Tooltip.mock.calls[0][0]
  expect(tooltip.formatter(0.68, 'Accuracy')).toEqual(['68.0%', 'Accuracy'])
  expect(tooltip.formatter(0.61, 'Loss')).toEqual([0.61, 'Loss'])
})

test('plots remaining model size and distinguishes false support from unreported support', () => {
  show('Core ML Optimizer', { apple_silicon: false, neural_engine_support: null })
  expect(points('bar-chart')).toEqual([{ metric: 'Model Size (%)', before: 100, after: 28 }])
  expect(screen.getByText('72%')).toBeInTheDocument()
  expect(screen.getByText('Apple Silicon').nextSibling).toHaveTextContent('Not available')
  expect(screen.getByText('Neural Engine').nextSibling).toHaveTextContent('Not reported')
  expect(screen.getByText('Quantization').nextSibling).toHaveTextContent('Available')
})

test('does not add synthetic baseline competitors or pipeline timings', () => {
  show('Baseline Model')
  expect(points('line-chart')).toEqual([{ model: 'LogisticRegression', accuracy: 0.982, validation: 0.957, f1: 0.924 }])
  expect(screen.getByText('Per-stage processing times were not supplied.')).toBeInTheDocument()
  const tooltip = Tooltip.mock.calls[0][0]
  expect(tooltip.formatter(0.924, 'F1-Score')).toEqual(['92.40%', 'F1-Score'])
})

test('labels similarity pairs from the reported fields', () => {
  show('Feature Extraction')
  expect(points('bar-chart').map(point => point.pair)).toEqual([
    'between-groups',
    'within-brighten',
    'within-darken',
    'similarity-score',
  ])
})

test('preserves similarity signs and zeros and omits unreported comparison pairs', () => {
  show('Feature Extraction', { between_group_similarity: 0, within_group_similarity_brighten: -0.5, within_group_similarity_darken: null })
  expect(points('bar-chart')).toEqual([
    { pair: 'between-groups', similarity: 0 },
    { pair: 'within-brighten', similarity: -0.5 },
    { pair: 'similarity-score', similarity: 0.87 }
  ])
  expect(screen.getByText('Embedding model not reported')).toBeInTheDocument()
  expect(screen.getByText('Method: Not reported')).toBeInTheDocument()
})
