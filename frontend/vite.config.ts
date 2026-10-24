import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { liveRunnerPlugin } from './live_runner';

export default defineConfig({
  plugins: [react(), liveRunnerPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: '127.0.0.1', // 只在本机监听（原来 host: true 对局域网开放，加上命令执行台等于远程执行）
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8780',
        changeOrigin: true,
      },
    },
  },
});
