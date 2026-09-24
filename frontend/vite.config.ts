import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_PROXY || 'http://127.0.0.1:8090';
  return {
    plugins: [react()],
    base: env.VITE_BASE_PATH || '/',
    server: {
      host: '127.0.0.1',
      port: 5180,
      proxy: Object.fromEntries(
        ['/v1', '/health', '/ready', '/metrics', '/docs', '/openapi.json'].map((path) => [
          path,
          target,
        ]),
      ),
    },
  };
});
