import axios from 'axios'

// Vite supplies the public API origin at build time. An empty origin uses the
// same-origin /api routes (and the Vite proxy during local development).
const baseURL = typeof __API_BASE_URL__ === 'undefined' ? '' : __API_BASE_URL__

export const createApiClient = (apiBaseURL = baseURL) => axios.create({
  baseURL: apiBaseURL,
  timeout: 30000,
  headers: { Accept: 'application/json' },
})

export default createApiClient()
