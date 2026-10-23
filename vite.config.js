import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  base: '/PicoTuri-EditJudge/',
  plugins: [react()],
  define: {
    __API_BASE_URL__: JSON.stringify(loadEnv(mode, process.cwd(), 'VITE_').VITE_API_BASE_URL || ''),
  },
  build: {
    chunkSizeWarningLimit: 1000,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      }
    }
  }
}))
