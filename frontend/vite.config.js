import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react({
      // Tableau Embedding API v3가 등록하는 <tableau-viz> 커스텀 엘리먼트를
      // React가 알 수 없는 DOM 엘리먼트로 경고하지 않도록 제외한다.
      include: '**/*.{jsx,tsx}',
    }),
  ],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
