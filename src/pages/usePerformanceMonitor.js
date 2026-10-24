import { useEffect, useState } from 'react'
import api from '../utils/api'

const validPerformanceData = (data) => {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return false

  const ranges = {
    cpu_usage: 100,
    memory_usage: 100,
    uptime: 100,
    error_rate: 1,
    requests_per_second: Infinity,
    average_response_time: Infinity,
    models_loaded: Infinity,
  }
  return Number.isInteger(data.models_loaded) && Object.entries(ranges).every(([key, maximum]) => (
    Number.isFinite(data[key]) && data[key] >= 0 && data[key] <= maximum
  ))
}

const pollIntervalMs = 5000

export function usePerformanceMonitor() {
  const [performanceData, setPerformanceData] = useState(null)
  const [realTimeData, setRealTimeData] = useState([])
  const [isLive, setIsLive] = useState(true)
  const [systemStatus, setSystemStatus] = useState('connecting')

  useEffect(() => {
    if (!isLive) return undefined

    let stopped = false
    let timerId = 0
    const controller = new AbortController()
    const clearPollTimer = () => clearTimeout(timerId)

    const poll = async () => {
      try {
        const response = await api.get('/api/performance/status', { signal: controller.signal })
        if (stopped) return
        if (!validPerformanceData(response.data)) throw new Error('Invalid performance data')

        const data = response.data
        setPerformanceData(data)
        setSystemStatus('online')
        setRealTimeData(previous => {
          const isDemo = data.data_source === 'demo'
          // A source change starts a new series so demo history is never
          // presented alongside measured samples without its demo label.
          const history = previous.at(-1)?.isDemo === isDemo ? previous : []
          return [...history.slice(-29), {
            timestamp: new Date().toLocaleTimeString(),
            cpu: data.cpu_usage,
            memory: data.memory_usage,
            requests_per_second: data.requests_per_second,
            isDemo,
          }]
        })
      } catch {
        if (!stopped) setSystemStatus('offline')
      } finally {
        // Schedule after completion so slow requests cannot overlap or reorder samples.
        if (!stopped) timerId = setTimeout(poll, pollIntervalMs)
      }
    }

    poll()
    return () => {
      stopped = true
      clearPollTimer()
      controller.abort()
    }
  }, [isLive])

  return { performanceData, realTimeData, isLive, setIsLive, systemStatus }
}

export function connectionStatus(isLive, systemStatus) {
  if (!isLive) {
    return {
      label: 'UPDATES PAUSED',
      color: 'border-gray-500/50 bg-gray-500/10 text-gray-300',
    }
  }
  if (systemStatus === 'online') {
    return {
      label: 'API CONNECTED',
      color: 'border-green-500/50 bg-green-500/10 text-green-400',
    }
  }
  if (systemStatus === 'offline') {
    return {
      label: 'API UNAVAILABLE',
      color: 'border-red-500/50 bg-red-500/10 text-red-400',
    }
  }
  return {
    label: 'CONNECTING',
    color: 'border-gray-500/50 bg-gray-500/10 text-gray-300',
  }
}

export function unavailableMessage(performanceData, isLive) {
  const stale = performanceData
    ? 'Performance data unavailable. Showing the last successful sample. '
    : 'Performance data unavailable. '
  return `${stale}${isLive ? 'Updates will retry automatically.' : 'Resume updates to retry.'}`
}

export function systemMetrics(performanceData) {
  if (!performanceData) return []
  return [
    { name: 'CPU Usage', value: performanceData.cpu_usage, unit: '%', color: '#3b82f6' },
    { name: 'Memory Usage', value: performanceData.memory_usage, unit: '%', color: '#f59e0b' },
    { name: 'Requests/sec', value: performanceData.requests_per_second, unit: '/s', color: '#10b981' },
    { name: 'Response Time', value: performanceData.average_response_time, unit: 'ms', color: '#8b5cf6' },
  ]
}
