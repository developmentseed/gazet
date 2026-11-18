import { defineConfig, type Plugin } from 'vite';

// wllama runs multithreaded only with cross-origin isolation. Every cross-origin request the page makes (Overture S3,
// the basemap, the model and data hosts) is a CORS request, which require-corp allows. Set on every response,
// including Vite's pre-bundled workers, which `server.headers` does not reach.
const isolation: Plugin = {
  name: 'cross-origin-isolation',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      next();
    });
  },
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      next();
    });
  },
};

export default defineConfig({
  base: './',
  plugins: [isolation],
  optimizeDeps: { exclude: ['@wllama/wllama', '@duckdb/duckdb-wasm', 'maplibre-gl'] },
  build: { target: 'es2022' },
});
