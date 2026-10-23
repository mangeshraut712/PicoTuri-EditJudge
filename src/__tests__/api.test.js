import axios from 'axios'
import api, { createApiClient } from '../utils/api'

test('uses same-origin API routes by default without changing global axios defaults', () => {
  expect(api.defaults.baseURL).toBe('')
  expect(api.getUri({ url: '/api/health' })).toBe('/api/health')
  expect(axios.defaults.baseURL).toBeUndefined()
  expect(api.defaults.timeout).toBe(30000)
})

test('routes requests to a configured deployment origin', () => {
  const remote = createApiClient('https://api.example.com/')
  expect(remote.getUri({ url: '/api/performance/status' })).toBe('https://api.example.com/api/performance/status')
  expect(api.defaults.baseURL).toBe('')
})

test('loads the build-time API origin when configured by Vite', () => {
  globalThis.__API_BASE_URL__ = 'https://api.example.com'
  try {
    jest.isolateModules(() => {
      const configured = require('../utils/api').default
      expect(configured.getUri({ url: '/api/health' })).toBe('https://api.example.com/api/health')
    })
  } finally {
    delete globalThis.__API_BASE_URL__
  }
})
