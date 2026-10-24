import { act, fireEvent, render, screen, within } from '@testing-library/react'
import api from '../utils/api'
import AlgorithmsPage from '../pages/AlgorithmsPage'
import { algorithmEndpoints, algorithmResults } from './algorithmFixtures'

jest.mock('../utils/api', () => ({ __esModule: true, default: { post: jest.fn() } }))
// The chart suite verifies rendering and numerical mappings separately.
jest.mock('../components/algorithms/ChartVisualization', () => ({
  ...jest.requireActual('../components/algorithms/ChartVisualization'),
  __esModule: true,
  default: ({ result }) => <div data-testid="result-data">{JSON.stringify(result)}</div>
}))

const card = name => within(screen.getByRole('heading', { name, exact: true }).parentElement)

beforeEach(() => { api.post.mockReset() })

test.each(Object.entries(algorithmEndpoints))('runs %s using its endpoint and displays the result', async (name, endpoint) => {
  api.post.mockResolvedValue({ data: algorithmResults[name] })
  render(<AlgorithmsPage />)
  fireEvent.click(card(name).getByRole('button', { name: 'Test Now' }))
  expect(await card(name).findByRole('button', { name: 'Tested' })).toBeEnabled()
  expect(api.post).toHaveBeenCalledWith(endpoint, undefined, { signal: expect.any(AbortSignal) })
  expect(screen.getByText('Completed').previousSibling).toHaveTextContent('1')
  fireEvent.click(card(name).getByRole('button', { name: 'View Results' }))
  expect(screen.getByRole('dialog', { name: `${name} results` })).toBeInTheDocument()
  expect(JSON.parse(screen.getByTestId('result-data').textContent)).toMatchObject(algorithmResults[name])
  fireEvent.click(screen.getByRole('button', { name: 'Close results' }))
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

test.each(Object.keys(algorithmResults))('labels offline %s fallback as demo data', async name => {
  api.post.mockRejectedValue(new Error('Network unavailable'))
  render(<AlgorithmsPage />)
  fireEvent.click(card(name).getByRole('button', { name: 'Test Now' }))
  expect(await card(name).findByText('Demo data')).toBeInTheDocument()
  expect(card(name).queryByText('Complete')).not.toBeInTheDocument()
  expect(screen.getByText('Completed').previousSibling).toHaveTextContent('0')
  expect(screen.getByText('Demo Results').previousSibling).toHaveTextContent('1')
  fireEvent.click(card(name).getByRole('button', { name: 'View Results' }))
  expect(screen.getByRole('status')).toHaveTextContent('API unavailable. Showing demo data.')
})

test('closes results from the keyboard, the backdrop, and keeps focus in the dialog', async () => {
  api.post.mockResolvedValue({ data: algorithmResults['Quality Scorer'] })
  render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  fireEvent.click(await card('Quality Scorer').findByRole('button', { name: 'View Results' }))
  const dialog = screen.getByRole('dialog', { name: 'Quality Scorer results' })
  const close = screen.getByRole('button', { name: 'Close results' })
  expect(close).toHaveFocus()
  fireEvent.keyDown(dialog, { key: 'Tab' })
  expect(close).toHaveFocus()
  fireEvent.keyDown(dialog, { key: 'Escape' })
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'View Results' }))
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss results' }))
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'View Results' }))
  fireEvent(
    screen.getByRole('dialog', { name: 'Quality Scorer results' }),
    new Event('cancel', { bubbles: true, cancelable: true })
  )
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

test('labels API-supplied demonstration results and allows another attempt', async () => {
  api.post.mockResolvedValueOnce({ data: { ...algorithmResults['Quality Scorer'], data_source: 'demo' } })
  api.post.mockResolvedValueOnce({ data: algorithmResults['Quality Scorer'] })
  render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  await card('Quality Scorer').findByText('Demo data')
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'View Results' }))
  expect(screen.getByRole('status')).toHaveTextContent('Demo data supplied by the API.')
  fireEvent.click(screen.getByRole('button', { name: 'Close results' }))
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Try Again' }))
  expect(await card('Quality Scorer').findByText('Complete')).toBeInTheDocument()
})

test.each([null, '<html>Not an API</html>', { success: false }, { success: true }, {
  ...algorithmResults['Quality Scorer'], overall_score: '91'
}])('rejects unsuccessful or malformed API results: %j', async data => {
  api.post.mockResolvedValue({ data })
  render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('unsuccessful or invalid result')
  expect(card('Quality Scorer').queryByRole('button', { name: 'View Results' })).not.toBeInTheDocument()
  expect(screen.getByText('Failed', { selector: '.text-sm' }).previousSibling).toHaveTextContent('1')
})

test('displays server failures instead of replacing them with successful mock data', async () => {
  api.post.mockRejectedValue({ response: { status: 503 } })
  render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('API request failed (503)')
  expect(card('Quality Scorer').getByRole('button', { name: 'Try Again' })).toBeEnabled()
  expect(screen.queryByText('Demo data')).not.toBeInTheDocument()
})

test('replaces previous success with a failure when a repeated test fails', async () => {
  api.post
    .mockResolvedValueOnce({ data: algorithmResults['Quality Scorer'] })
    .mockRejectedValueOnce({ response: { status: 500 } })
  render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  fireEvent.click(await card('Quality Scorer').findByRole('button', { name: 'Tested' }))

  expect(await screen.findByRole('alert')).toHaveTextContent('API request failed (500)')
  expect(screen.getByText('Completed').previousSibling).toHaveTextContent('0')
  expect(card('Quality Scorer').queryByRole('button', { name: 'View Results' })).not.toBeInTheDocument()
})

test('keeps concurrent algorithm results separate when requests finish out of order', async () => {
  let finishQuality
  api.post
    .mockImplementationOnce(() => new Promise(resolve => { finishQuality = resolve }))
    .mockResolvedValueOnce({ data: algorithmResults['DPO Training'] })
  render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  fireEvent.click(card('DPO Training').getByRole('button', { name: 'Test Now' }))
  expect(await card('DPO Training').findByRole('button', { name: 'Tested' })).toBeEnabled()
  expect(card('Quality Scorer').getByRole('button', { name: 'Testing...' })).toBeDisabled()

  await act(async () => { finishQuality({ data: algorithmResults['Quality Scorer'] }) })
  expect(card('Quality Scorer').getByRole('button', { name: 'Tested' })).toBeEnabled()
  expect(screen.getByText('Completed').previousSibling).toHaveTextContent('2')
})

test('keeps a request pending once and aborts it when the page unmounts', async () => {
  let resolveRequest
  api.post.mockImplementation(() => new Promise(resolve => { resolveRequest = resolve }))
  const { unmount } = render(<AlgorithmsPage />)
  fireEvent.click(card('Quality Scorer').getByRole('button', { name: 'Test Now' }))
  const pending = card('Quality Scorer').getByRole('button', { name: 'Testing...' })
  expect(pending).toBeDisabled()
  fireEvent.click(pending)
  expect(api.post).toHaveBeenCalledTimes(1)
  const signal = api.post.mock.calls[0][2].signal
  unmount()
  expect(signal.aborted).toBe(true)
  await act(async () => resolveRequest({ data: algorithmResults['Quality Scorer'] }))
})
