import { StrictMode } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import PerformanceDashboard from '../pages/PerformanceDashboard'
import api from '../utils/api'

jest.mock('../utils/api', () => ({ __esModule: true, default: { get: jest.fn() } }))
jest.mock('recharts', () => {
  const React = require('react')
  const Container = ({ children }) => React.createElement('div', null, children)
  const Empty = () => null
  return {
    ResponsiveContainer: Container,
    AreaChart: ({ data }) => React.createElement('div', {
      'data-testid': 'performance-samples',
      'data-samples': JSON.stringify(data),
    }),
    PieChart: Container,
    Pie: Empty,
    Cell: Empty,
    Area: Empty,
    XAxis: Empty,
    YAxis: Empty,
    CartesianGrid: Empty,
    Tooltip: Empty,
  }
})

const sample = (changes = {}) => ({
  cpu_usage: 42,
  memory_usage: 58,
  requests_per_second: 23,
  average_response_time: 95,
  uptime: 99,
  error_rate: 0.015,
  models_loaded: 3,
  ...changes,
})
const samples = () => JSON.parse(screen.getByTestId('performance-samples').dataset.samples)
const settle = async () => { await act(async () => {}) }
const tick = async (milliseconds = 5000) => {
  await act(async () => { jest.advanceTimersByTime(milliseconds) })
}
const deferred = () => {
  let resolve
  let reject
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve
    reject = onReject
  })
  return { promise, resolve, reject }
}

beforeEach(() => {
  jest.useFakeTimers()
  api.get.mockReset()
})

afterEach(() => {
  jest.clearAllTimers()
  jest.useRealTimers()
})

test('shows connecting without inventing samples while the first request is pending', () => {
  api.get.mockReturnValue(new Promise(() => {}))
  render(<PerformanceDashboard />)

  expect(screen.getByRole('status')).toHaveTextContent('CONNECTING')
  expect(samples()).toEqual([])
  expect(screen.queryByText('CPU Usage')).not.toBeInTheDocument()
  expect(screen.getAllByText('—')).toHaveLength(3)
  expect(screen.getByText('Sample Algorithm Performance Metrics')).toBeInTheDocument()
})

test('appends actual samples instead of resetting history or inventing intermediate readings', async () => {
  api.get
    .mockResolvedValueOnce({ data: sample({ cpu_usage: 10 }) })
    .mockResolvedValueOnce({ data: sample({ cpu_usage: 20 }) })
  render(<PerformanceDashboard />)
  await settle()

  expect(screen.getByRole('status')).toHaveTextContent('API CONNECTED')
  expect(samples().map(point => point.cpu)).toEqual([10])
  await tick(4000)
  expect(samples().map(point => point.cpu)).toEqual([10])
  await tick(1000)
  expect(samples().map(point => point.cpu)).toEqual([10, 20])
  expect(samples().map(point => point.memory)).toEqual([58, 58])
})

test('retains only the most recent 30 successful samples', async () => {
  let cpu = 0
  api.get.mockImplementation(() => Promise.resolve({ data: sample({ cpu_usage: cpu++ }) }))
  render(<PerformanceDashboard />)
  await settle()
  for (let index = 0; index < 31; index += 1) await tick()

  expect(samples()).toHaveLength(30)
  expect(samples().map(point => point.cpu)).toEqual(Array.from({ length: 30 }, (_, index) => index + 2))
})

test('does not overlap slow requests and waits five seconds after completion', async () => {
  const request = deferred()
  api.get.mockReturnValue(request.promise)
  render(<PerformanceDashboard />)
  await tick(20000)
  expect(api.get).toHaveBeenCalledTimes(1)

  await act(async () => { request.resolve({ data: sample() }) })
  await tick(4999)
  expect(api.get).toHaveBeenCalledTimes(1)
  await tick(1)
  expect(api.get).toHaveBeenCalledTimes(2)
})

test('pausing aborts in-flight updates and ignores their late responses after resuming', async () => {
  const oldRequest = deferred()
  api.get
    .mockReturnValueOnce(oldRequest.promise)
    .mockResolvedValueOnce({ data: sample({ cpu_usage: 25 }) })
  render(<PerformanceDashboard />)
  const oldSignal = api.get.mock.calls[0][1].signal

  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }))
  expect(oldSignal.aborted).toBe(true)
  expect(screen.getByRole('status')).toHaveTextContent('UPDATES PAUSED')
  await tick(15000)
  expect(api.get).toHaveBeenCalledTimes(1)

  fireEvent.click(screen.getByRole('button', { name: 'Resume updates' }))
  await settle()
  expect(samples().map(point => point.cpu)).toEqual([25])
  await act(async () => { oldRequest.resolve({ data: sample({ cpu_usage: 99 }) }) })
  expect(samples().map(point => point.cpu)).toEqual([25])
  await tick()
  expect(api.get).toHaveBeenCalledTimes(3)
})

test('pausing preserves already received samples and cancels the next scheduled poll', async () => {
  api.get.mockResolvedValue({ data: sample() })
  render(<PerformanceDashboard />)
  await settle()
  const savedSamples = samples()
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }))
  await tick(15000)

  expect(api.get).toHaveBeenCalledTimes(1)
  expect(samples()).toEqual(savedSamples)
})

test('shows unavailable metrics on failure and recovers on the next poll', async () => {
  api.get
    .mockRejectedValueOnce(new Error('Network error'))
    .mockResolvedValueOnce({ data: sample() })
  render(<PerformanceDashboard />)
  await settle()

  expect(screen.getByRole('status')).toHaveTextContent('API UNAVAILABLE')
  expect(screen.getByRole('alert')).toHaveTextContent('Performance data unavailable')
  expect(samples()).toEqual([])
  expect(screen.getAllByText('—')).toHaveLength(3)

  await tick()
  expect(screen.getByRole('status')).toHaveTextContent('API CONNECTED')
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(samples()).toHaveLength(1)
})

test('marks retained metrics as stale after failure without appending fake samples', async () => {
  api.get
    .mockResolvedValueOnce({ data: sample({ cpu_usage: 13 }) })
    .mockRejectedValueOnce(new Error('Timeout'))
  render(<PerformanceDashboard />)
  await settle()
  await tick()

  expect(screen.getByRole('alert')).toHaveTextContent('Showing the last successful sample')
  expect(screen.getByText('13.0')).toBeInTheDocument()
  expect(samples().map(point => point.cpu)).toEqual([13])
})

test('does not promise automatic retries while failed updates are paused', async () => {
  api.get.mockRejectedValue(new Error('Network unavailable'))
  render(<PerformanceDashboard />)
  await settle()
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }))

  expect(screen.getByRole('alert')).toHaveTextContent('Resume updates to retry.')
  expect(screen.getByRole('alert')).not.toHaveTextContent('retry automatically')
  await tick(15000)
  expect(api.get).toHaveBeenCalledTimes(1)
})

test.each([
  null,
  {},
  '<html>Not an API response</html>',
  sample({ cpu_usage: '42' }),
  sample({ cpu_usage: NaN }),
  sample({ memory_usage: Infinity }),
  sample({ cpu_usage: 101 }),
  sample({ memory_usage: -1 }),
  sample({ error_rate: 1.1 }),
  sample({ models_loaded: {} }),
  sample({ models_loaded: 1.5 }),
])('rejects malformed metrics without crashing or claiming a successful update: %p', async data => {
  api.get.mockResolvedValue({ data })
  render(<PerformanceDashboard />)
  await settle()

  expect(screen.getByRole('status')).toHaveTextContent('API UNAVAILABLE')
  expect(samples()).toEqual([])
  expect(screen.queryByText('CPU Usage')).not.toBeInTheDocument()
})

test('preserves valid zeros in every displayed system metric', async () => {
  api.get.mockResolvedValue({ data: sample({
    cpu_usage: 0,
    memory_usage: 0,
    requests_per_second: 0,
    average_response_time: 0,
    uptime: 0,
    error_rate: 0,
    models_loaded: 0,
  }) })
  render(<PerformanceDashboard />)
  await settle()

  expect(screen.getAllByText('0.0')).toHaveLength(4)
  expect(screen.getByText('System Uptime').parentElement).toHaveTextContent('0%')
  expect(screen.getByText('Error Rate').parentElement).toHaveTextContent('0.00%')
  expect(screen.getByText('Active Models').parentElement).toHaveTextContent('0')
  expect(samples()[0]).toMatchObject({ cpu: 0, memory: 0, requests_per_second: 0 })
})

test('identifies simulated backend metrics as demo data', async () => {
  api.get.mockResolvedValue({ data: sample({ data_source: 'demo' }) })
  render(<PerformanceDashboard />)
  await settle()

  expect(screen.getByText(/these samples are simulated/)).toBeInTheDocument()
})

test('starts a separate history when switching between demo and measured samples', async () => {
  api.get
    .mockResolvedValueOnce({ data: sample({ data_source: 'demo', cpu_usage: 10 }) })
    .mockResolvedValueOnce({ data: sample({ cpu_usage: 20 }) })
    .mockResolvedValueOnce({ data: sample({ data_source: 'demo', cpu_usage: 30 }) })
  render(<PerformanceDashboard />)
  await settle()

  expect(samples().map(point => point.cpu)).toEqual([10])
  expect(screen.getByText(/these samples are simulated/)).toBeInTheDocument()
  await tick()
  expect(samples().map(point => point.cpu)).toEqual([20])
  expect(screen.queryByText(/these samples are simulated/)).not.toBeInTheDocument()
  await tick()
  expect(samples().map(point => point.cpu)).toEqual([30])
  expect(screen.getByText(/these samples are simulated/)).toBeInTheDocument()
})

test('clearly labels static algorithm counts and success rates as demonstration values', () => {
  api.get.mockReturnValue(new Promise(() => {}))
  render(<PerformanceDashboard />)

  expect(screen.getByText('Demo request distribution; these counts are illustrative.')).toBeInTheDocument()
  expect(screen.getByText('Demonstration values, not measured algorithm results.')).toBeInTheDocument()
  expect(screen.getAllByRole('row')).toHaveLength(8)
  expect(screen.queryByRole('columnheader', { name: 'Uptime' })).not.toBeInTheDocument()
  expect(screen.queryByText('99.9%')).not.toBeInTheDocument()
})

test('formats uptime to one decimal place', async () => {
  api.get.mockResolvedValue({ data: sample({ uptime: 99.126 }) })
  render(<PerformanceDashboard />)
  await settle()
  expect(screen.getByText('System Uptime').parentElement).toHaveTextContent('99.1%')
})

test('aborts requests on unmount and never schedules another poll', async () => {
  const request = deferred()
  api.get.mockReturnValue(request.promise)
  const { unmount } = render(<PerformanceDashboard />)
  const signal = api.get.mock.calls[0][1].signal
  unmount()

  expect(signal.aborted).toBe(true)
  await act(async () => { request.reject(new Error('Canceled')) })
  await tick(15000)
  expect(api.get).toHaveBeenCalledTimes(1)
})

test('ignores the discarded effect request under React StrictMode', async () => {
  const staleRequest = deferred()
  api.get
    .mockReturnValueOnce(staleRequest.promise)
    .mockResolvedValueOnce({ data: sample({ cpu_usage: 30 }) })
  render(<StrictMode><PerformanceDashboard /></StrictMode>)
  await settle()

  expect(api.get.mock.calls[0][1].signal.aborted).toBe(true)
  expect(samples().map(point => point.cpu)).toEqual([30])
  await act(async () => { staleRequest.resolve({ data: sample({ cpu_usage: 90 }) }) })
  expect(samples().map(point => point.cpu)).toEqual([30])
})
