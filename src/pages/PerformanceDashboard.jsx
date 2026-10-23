import React, { useState, useEffect } from 'react'
import { Activity, Brain, BarChart3, LineChart as LineChartIcon, AlertTriangle, CheckCircle, Server, Database } from 'lucide-react'
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell } from 'recharts'
import api from '../utils/api'

const validPerformanceData = (data) => {
  if (!data || typeof data !== 'object') return false

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

const PerformanceDashboard = () => {
  const [performanceData, setPerformanceData] = useState(null)
  const [realTimeData, setRealTimeData] = useState([])
  const [isLive, setIsLive] = useState(true)
  const [systemStatus, setSystemStatus] = useState('connecting')
  const statusLabel = !isLive ? 'UPDATES PAUSED' : {
    connecting: 'CONNECTING',
    online: 'API CONNECTED',
    offline: 'API UNAVAILABLE',
  }[systemStatus]
  const statusColor = !isLive || systemStatus === 'connecting'
    ? 'border-gray-500/50 bg-gray-500/10 text-gray-300'
    : systemStatus === 'online'
      ? 'border-green-500/50 bg-green-500/10 text-green-400'
      : 'border-red-500/50 bg-red-500/10 text-red-400'

  useEffect(() => {
    if (!isLive) return

    let stopped = false
    let timer
    const controller = new AbortController()

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
        if (!stopped) timer = setTimeout(poll, 5000)
      }
    }

    poll()
    return () => {
      stopped = true
      clearTimeout(timer)
      controller.abort()
    }
  }, [isLive])

  // Create statistics for the system
  const systemMetrics = performanceData ? [
    { name: 'CPU Usage', value: performanceData.cpu_usage, unit: '%', color: '#3b82f6' },
    { name: 'Memory Usage', value: performanceData.memory_usage, unit: '%', color: '#f59e0b' },
    { name: 'Requests/sec', value: performanceData.requests_per_second, unit: '/s', color: '#10b981' },
    { name: 'Response Time', value: performanceData.average_response_time, unit: 'ms', color: '#8b5cf6' }
  ] : []

  // Mock data for algorithm performance
  const algorithmPerformance = [
    { name: 'Quality Scorer', calls: 1247, success: 99.1, avgTime: 28.6, color: '#3b82f6' },
    { name: 'Diffusion Model', calls: 856, success: 97.8, avgTime: 83.2, color: '#10b981' },
    { name: 'DPO Training', calls: 234, success: 95.7, avgTime: 1024, color: '#f59e0b' },
    { name: 'Multi-turn Editor', calls: 589, success: 98.3, avgTime: 148.3, color: '#8b5cf6' },
    { name: 'Core ML', calls: 342, success: 99.7, avgTime: 45.2, color: '#ec4899' },
    { name: 'Baseline', calls: 987, success: 100, avgTime: 12.3, color: '#6b7280' },
    { name: 'Feature Extraction', calls: 1456, success: 98.9, avgTime: 8.7, color: '#14b8a6' }
  ]

  // Prepare data for pie chart
  const pieData = algorithmPerformance.map(algo => ({
    name: algo.name,
    value: algo.calls,
    fill: algo.color
  }))

  return (
    <div className="lg:ml-80 min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="text-center">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4 flex items-center justify-center gap-4">
              <Activity className="w-12 h-12" />
              Performance Monitoring
            </h1>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              System performance samples and example algorithm metrics
            </p>

            {/* System Status Indicator */}
            <div className="mt-6 flex justify-center">
              <div role="status" className={`px-4 py-2 rounded-full glass border flex items-center gap-2 ${statusColor}`}>
                <div className={`w-3 h-3 rounded-full bg-current ${isLive ? 'animate-pulse' : ''}`}></div>
                <span className="text-sm font-bold">
                  {statusLabel}
                </span>
              </div>
            </div>
            {performanceData?.data_source === 'demo' && (
              <p className="mt-4 text-amber-300">Demo data: these samples are simulated, not measured system performance.</p>
            )}
            {systemStatus === 'offline' && (
              <p role="alert" className="mt-4 text-red-300">
                {performanceData
                  ? 'Performance data unavailable. Showing the last successful sample. '
                  : 'Performance data unavailable. '}
                {isLive ? 'Updates will retry automatically.' : 'Resume updates to retry.'}
              </p>
            )}
          </div>
        </div>

        {/* System Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {systemMetrics.map((metric, index) => (
            <MetricCard
              key={metric.name}
              title={metric.name}
              value={metric.value}
              unit={metric.unit}
              color={metric.color}
              delay={index * 100}
            />
          ))}
        </div>

        {/* Main Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Real-time Performance */}
          <div className="glass rounded-3xl p-8">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                  <LineChartIcon className="w-6 h-6 text-blue-400" />
                  Performance Samples
                </h2>
                <p className="text-gray-400">One sample per successful API update</p>
              </div>
              <button
                onClick={() => setIsLive(previous => !previous)}
                aria-label={isLive ? 'Pause updates' : 'Resume updates'}
                aria-pressed={isLive}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  isLive
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
                }`}
              >
                {isLive ? 'Pause' : 'Resume'}
              </button>
            </div>

            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={realTimeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="cpu"
                  stackId="1"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                  name="CPU %"
                />
                <Area
                  type="monotone"
                  dataKey="memory"
                  stackId="2"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.3}
                  name="Memory %"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Algorithm Distribution */}
          <div className="glass rounded-3xl p-8">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                  <BarChart3 className="w-6 h-6 text-purple-400" />
                  Algorithm Usage
                </h2>
                <p className="text-gray-400">Demo request distribution; these counts are illustrative.</p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(17, 24, 39, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                    formatter={(value, name) => [`${value} calls`, name]}
                  />
                </PieChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-2 gap-3">
                {algorithmPerformance.slice(0, 4).map(algo => (
                  <div key={algo.name} className="flex items-center justify-between p-3 glass rounded-lg">
                    <div>
                      <div className="text-sm font-semibold text-white">{algo.name}</div>
                      <div className="text-xs text-gray-400">{algo.calls} calls</div>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
                      algo.success >= 99 ? 'bg-green-500/20 text-green-400' :
                      algo.success >= 95 ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      <CheckCircle className="w-3 h-3" />
                      {algo.success.toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Algorithm Performance Table */}
        <div className="glass rounded-3xl p-8">
          <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
            <Brain className="w-6 h-6 text-purple-400" />
            Sample Algorithm Performance Metrics
          </h2>
          <p className="text-amber-300 mb-6">Demonstration values, not measured algorithm results.</p>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-4 pr-6 text-gray-400 font-medium">Algorithm</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">Total Calls</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">Success Rate</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">Avg Response</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">Uptime</th>
                </tr>
              </thead>
              <tbody>
                {algorithmPerformance.map(algo => (
                  <tr key={algo.name} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td className="py-4 pr-6">
                      <div className="flex items-center gap-3">
                        <div
                          className="w-3 h-3 rounded-full animate-pulse"
                          style={{ backgroundColor: algo.color }}
                        ></div>
                        <span className="font-medium text-white">{algo.name}</span>
                      </div>
                    </td>
                    <td className="text-right py-4 px-6 text-white font-mono">
                      {algo.calls.toLocaleString()}
                    </td>
                    <td className="text-right py-4 px-6">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        algo.success >= 99 ? 'bg-green-500/20 text-green-400' :
                        algo.success >= 95 ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {algo.success.toFixed(1)}%
                      </span>
                    </td>
                    <td className="text-right py-4 px-6 text-white font-mono">
                      {algo.avgTime.toFixed(1)}ms
                    </td>
                    <td className="text-right py-4 px-6">
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400">
                        99.9%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* System Health Footer */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <StatusCard
            title="System Uptime"
            value={performanceData ? `${performanceData.uptime}%` : '—'}
            icon={<Server className="w-5 h-5" />}
          />
          <StatusCard
            title="Active Models"
            value={performanceData?.models_loaded ?? '—'}
            icon={<Database className="w-5 h-5" />}
          />
          <StatusCard
            title="Error Rate"
            value={performanceData ? `${(performanceData.error_rate * 100).toFixed(2)}%` : '—'}
            icon={<AlertTriangle className="w-5 h-5" />}
          />
        </div>
      </div>
    </div>
  )
}

const MetricCard = ({ title, value, unit, color, delay = 0 }) => (
  <div
    className="glass rounded-2xl p-6 hover:scale-105 transition-all duration-300 card-hover"
    style={{ animationDelay: `${delay}ms` }}
  >
    <div className="flex items-center justify-between mb-4">
      <div className="text-sm font-medium text-gray-400">{title}</div>
      <div
        className={`w-3 h-3 rounded-full animate-pulse`}
        style={{ backgroundColor: color }}
      ></div>
    </div>

    <div className="flex items-end gap-2">
      <div
        className="text-3xl font-bold"
        style={{ color }}
      >
        {value.toFixed(1)}
      </div>
      <div className="text-sm text-gray-400 pb-1">{unit}</div>
    </div>

  </div>
)

const StatusCard = ({ title, value, icon }) => (
  <div className="glass rounded-xl p-6 text-center">
    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full mb-4 bg-blue-500/20">
      {icon}
    </div>
    <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
    <p className="text-2xl font-bold text-blue-400">
      {value}
    </p>
  </div>
)

export default PerformanceDashboard
