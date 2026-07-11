import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  fullyParallel: false,
  retries: 0,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:8000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  webServer: {
    command: '.\\.venv\\Scripts\\python.exe -m uvicorn dubsync.web.app:create_app --factory --host 127.0.0.1 --port 8000',
    cwd: '..',
    url: 'http://127.0.0.1:8000/api/health',
    reuseExistingServer: true,
    timeout: 30_000,
    env: {
      DUBSYNC_DATA_DIR: 'runtime-data/e2e',
      DUBSYNC_PROVIDERS_PATH: 'web/e2e/fixtures/providers.yaml',
      DUBSYNC_STYLE_PATH: 'style_profile.yaml',
      DUBSYNC_STATIC_DIR: 'web/dist',
      DUBSYNC_PROCESSING_INLINE: '1',
      DUBSYNC_MAX_JOBS_PER_HOUR: '50',
    },
  },
})
