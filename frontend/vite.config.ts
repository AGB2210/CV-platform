import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

export default defineConfig({
  plugins: [react(), tailwindcss()],

  resolve: {
    // Lets us write `import { api } from '@/lib/api'` instead of counting
    // '../../..' segments. Mirrored in tsconfig.app.json so the editor agrees.
    alias: { '@': path.resolve(__dirname, './src') },
  },

  server: {
    port: 5173,

    // Proxy: any request the frontend makes to /api/* is forwarded by the Vite
    // dev server to the FastAPI process on :8000.
    //
    // Why bother, given the backend already sets CORS headers? Because it means
    // frontend code can call fetch('/api/health') with a relative URL — no
    // hardcoded hostname, no environment switch between dev and production.
    // In production the same relative path works when FastAPI serves the built
    // frontend from its own origin. The proxy makes dev behave like prod.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/static': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
