import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'

import * as api from '../api'
import { defaultConfig, type PublicConfig } from '../types'
import { Workspace } from './Workspace'

vi.mock('../api', async (original) => ({ ...await original<typeof import('../api')>(), createJob: vi.fn() }))

const config: PublicConfig = {
  ...defaultConfig, jobs_available: true,
  transcription_models: [
    { id: 'microsoft/mai-transcribe-2', label: 'MAI-Transcribe 2', available: true },
    { id: 'scribe_v2', label: 'Scribe v2', available: true },
  ],
  asr_cross_check_available: true,
  asr_cross_check_hourly_usd: { scribe_v2: 0.22, 'microsoft/mai-transcribe-2': 0.10 },
}

afterEach(() => { vi.clearAllMocks(); sessionStorage.clear() })

it('keeps additional checks collapsed and explains the extra cost before selection', async () => {
  const user = userEvent.setup()
  render(<Workspace config={config} />)
  const disclosure = screen.getByRole('button', { name: /additional checks/i })
  expect(disclosure).toHaveAttribute('aria-expanded', 'false')
  expect(screen.queryByRole('checkbox', { name: /cross-check wording/i })).not.toBeInTheDocument()

  disclosure.focus()
  await user.keyboard('{Enter}')
  expect(disclosure).toHaveAttribute('aria-expanded', 'true')
  const toggle = screen.getByRole('checkbox', { name: /cross-check wording/i })
  expect(toggle).not.toBeChecked()
  expect(toggle).toHaveAccessibleDescription(/Estimated extra transcription cost: about \$0\.22 per audio hour/)

  await user.click(toggle)
  await user.click(disclosure)
  expect(disclosure).toHaveAttribute('aria-expanded', 'false')
  expect(disclosure).toHaveAccessibleName(/additional checks.*on/i)
  expect(screen.queryByRole('checkbox', { name: /cross-check wording/i })).not.toBeInTheDocument()
  await user.click(disclosure)
  expect(screen.getByRole('checkbox', { name: /cross-check wording/i })).toBeChecked()
})

it.each([false, true])('submits an explicit cross-check preference: %s', async (enabled) => {
  vi.mocked(api.createJob).mockResolvedValue({ id: 'job', mode: 'sync', status: 'complete', progress: 100,
    expires_at: '2027-01-01', error: null, result: null, downloads: [] })
  const user = userEvent.setup()
  render(<Workspace config={config} />)
  await user.click(screen.getByRole('button', { name: /additional checks/i }))
  const toggle = screen.getByRole('checkbox', { name: /cross-check wording/i })
  expect(toggle).not.toBeChecked()
  expect(screen.getByText(/\$0\.22 per audio hour/)).toBeInTheDocument()
  if (enabled) await user.click(toggle)
  await user.upload(screen.getByLabelText('Dialogue audio'), new File(['audio'], '001.wav', { type: 'audio/wav' }))
  await user.upload(screen.getByLabelText('Original SRT'), new File(['subtitle'], '001.srt', { type: 'text/plain' }))
  await user.click(screen.getByRole('button', { name: 'Start sync' }))
  await waitFor(() => expect(api.createJob).toHaveBeenCalled())
  expect(vi.mocked(api.createJob).mock.calls[0][0].get('asr_cross_check')).toBe(String(enabled))
})

it('offers the other model, hides the toggle in generation, and resets it', async () => {
  const user = userEvent.setup()
  render(<Workspace config={config} />)
  await user.selectOptions(screen.getByLabelText('Transcription model'), 'scribe_v2')
  await user.click(screen.getByRole('button', { name: /additional checks/i }))
  expect(screen.getByText(/\$0\.10 per audio hour/)).toBeInTheDocument()
  await user.click(screen.getByRole('checkbox', { name: /cross-check wording/i }))
  await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
  expect(screen.queryByRole('checkbox', { name: /cross-check wording/i })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /additional checks/i })).not.toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Sync existing SRT' }))
  expect(screen.getByRole('button', { name: /additional checks/i })).toHaveAttribute('aria-expanded', 'false')
  await user.click(screen.getByRole('button', { name: /additional checks/i }))
  expect(screen.getByRole('checkbox', { name: /cross-check wording/i })).not.toBeChecked()
})

it('disables cross-check when a second provider is unavailable', async () => {
  const user = userEvent.setup()
  render(<Workspace config={{ ...config, asr_cross_check_available: false }} />)
  await user.click(screen.getByRole('button', { name: /additional checks/i }))
  expect(screen.getByRole('checkbox', { name: /cross-check wording/i })).toBeDisabled()
})
