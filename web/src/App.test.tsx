import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'

const configResponse = {
  retention_hours: 24,
  max_upload_bytes: 2_147_483_648,
  audio_extensions: ['.mp3', '.wav'],
  fps_values: [24, 25, 30],
  pricing: {
    generate: { usd_per_minute: 0.12, minimum_usd: 3 },
    sync: { usd_per_minute: 0.18, minimum_usd: 5 },
    precision: { usd_per_minute: 0.25, minimum_usd: 10 },
  },
  billing_enabled: false,
}

afterEach(() => {
  vi.restoreAllMocks()
  sessionStorage.clear()
})

describe('DubSync workspace', () => {
  it('shows the real sync workflow and requires both source files by default', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(configResponse), { status: 200 }))
    render(<App />)

    expect(screen.getByRole('heading', { name: 'Sync dialogue. Keep every cue honest.' })).toBeVisible()
    expect(screen.getByLabelText('Dialogue audio')).toBeRequired()
    expect(screen.getByLabelText('Original SRT')).toBeRequired()
    expect(screen.getByRole('button', { name: 'Start sync' })).toBeDisabled()
    expect(await screen.findByText('Files are deleted after 24 hours')).toBeVisible()
  })

  it('switches to audio-only mode and submits a generate job without an SRT', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: 'job-1',
          token: 'secret-token',
          mode: 'generate',
          status: 'complete',
          progress: 100,
          result: { cue_count: 12, cost_usd: 0.02 },
          downloads: ['srt', 'qc-json'],
          expires_at: '2026-07-12T00:00:00Z',
        }),
        { status: 202 },
      ),
    )

    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    const audio = new File(['audio'], 'episode.wav', { type: 'audio/wav' })
    await user.upload(screen.getByLabelText('Dialogue audio'), audio)
    const submit = screen.getByRole('button', { name: 'Generate SRT' })
    expect(submit).toBeEnabled()
    await user.click(submit)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    const [, options] = fetchMock.mock.calls[1]
    expect((options?.body as FormData).get('mode')).toBe('generate')
    expect((options?.body as FormData).get('subtitle')).toBeNull()
    expect(await screen.findByText('12 cues ready')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Download SRT' })).toBeEnabled()
  })

  it('restores an in-progress job after a page refresh', async () => {
    sessionStorage.setItem('dubsync:active-job', JSON.stringify({ id: 'restored-job', token: 'restored-token' }))
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, options) => {
      const url = String(input)
      if (url === '/api/config') return new Response(JSON.stringify(configResponse), { status: 200 })
      expect(url).toBe('/api/jobs/restored-job')
      expect((options?.headers as Record<string, string>).Authorization).toBe('Bearer restored-token')
      return new Response(
        JSON.stringify({
          id: 'restored-job',
          mode: 'sync',
          status: 'complete',
          progress: 100,
          result: { cue_count: 27, cost_usd: 0.04 },
          downloads: ['srt'],
          expires_at: '2026-07-12T00:00:00Z',
          error: null,
        }),
        { status: 200 },
      )
    })

    render(<App />)

    expect(await screen.findByText('27 cues ready')).toBeVisible()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('clears unusable restored access and shows the refresh error', async () => {
    sessionStorage.setItem('dubsync:active-job', JSON.stringify({ id: 'expired-job', token: 'expired-token' }))
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      if (String(input) === '/api/config') return new Response(JSON.stringify(configResponse), { status: 200 })
      return new Response('', { status: 404 })
    })

    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not refresh the job.')
    expect(sessionStorage.getItem('dubsync:active-job')).toBeNull()
  })

  it('shows API validation errors and permits another submission', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Audio could not be decoded.' }), { status: 422 }))
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    await user.upload(screen.getByLabelText('Dialogue audio'), new File(['audio'], 'broken.wav', { type: 'audio/wav' }))
    await user.selectOptions(screen.getByLabelText('Frame rate'), '25')
    await user.selectOptions(screen.getByLabelText('Language'), 'fr')
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Audio could not be decoded.')
    expect(screen.getByRole('button', { name: 'Generate SRT' })).toBeEnabled()
  })

  it('downloads a completed SRT with its bearer token', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      id: 'job-download', token: 'download-token', mode: 'generate', status: 'complete', progress: 100,
      result: { cue_count: 3, cost_usd: 0.01 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
    }), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(new Blob(['1\n00:00:00,000 --> 00:00:01,000\nHello\n']), {
      status: 200,
      headers: { 'content-disposition': 'attachment; filename="result.srt"' },
    }))
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    await user.upload(screen.getByLabelText('Dialogue audio'), new File(['audio'], 'episode.wav', { type: 'audio/wav' }))
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))
    await user.click(await screen.findByRole('button', { name: 'Download SRT' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    expect(fetchMock.mock.calls[2][1]?.headers).toEqual({ Authorization: 'Bearer download-token' })
  })
})
