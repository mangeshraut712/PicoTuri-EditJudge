import axios from 'axios'

// Configure axios defaults
axios.defaults.baseURL = import.meta.env.PROD ? 'http://localhost:5001' : 'http://localhost:5001'
axios.defaults.timeout = 30000 // 30 seconds timeout
axios.defaults.headers.common['Content-Type'] = 'application/json'

// Add request interceptor for debugging
axios.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('❌ Request Error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for error handling
axios.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('❌ API Error:', error.response?.status, error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default axios
