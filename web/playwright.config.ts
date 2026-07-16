import { defineConfig } from '@playwright/test'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const e2eDataDir = join(tmpdir(), `dubsync-e2e-${process.pid}-${Date.now()}`)

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
      DUBSYNC_DATA_DIR: e2eDataDir,
      DUBSYNC_PROVIDERS_PATH: 'web/e2e/fixtures/providers.yaml',
      DUBSYNC_STYLE_PATH: 'style_profile.yaml',
      DUBSYNC_STATIC_DIR: 'web/dist',
      DUBSYNC_PROCESSING_INLINE: '0',
      DUBSYNC_MAX_SUBMISSIONS_PER_HOUR: '50',
      DUBSYNC_REQUIRE_JOB_ACCESS_CODE: '0',
      DUBSYNC_JOB_ACCESS_CODE: 'fixture-access-code',
    },
  },
})
