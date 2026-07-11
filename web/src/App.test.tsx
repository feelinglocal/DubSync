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
  access_code_required: false,
  jobs_available: true,
  generation_styles: {
    default_preset: 'standard',
    presets: [
      {
        id: 'standard',
        name: 'DubSync default',
        values: {
          max_lines_per_cue: 2, max_chars_per_line: 26,
          min_cue_duration_seconds: 0.5, max_cue_duration_seconds: 5,
          min_cps: 2, max_cps: 30, max_gap_seconds: 0.8, lead_in_ms: 0, tail_ms: 40,
        },
      },
      {
        id: 'streaming',
        name: 'Streaming',
        values: {
          max_lines_per_cue: 2, max_chars_per_line: 42,
          min_cue_duration_seconds: 1, max_cue_duration_seconds: 7,
          min_cps: 2, max_cps: 20, max_gap_seconds: 1, lead_in_ms: 0, tail_ms: 120,
        },
      },
    ],
    custom_limits: {
      max_lines_per_cue: { min: 1, max: 4, step: 1 },
      max_chars_per_line: { min: 10, max: 80, step: 1 },
      min_cue_duration_seconds: { min: 0.2, max: 5, step: 0.1 },
      max_cue_duration_seconds: { min: 0.5, max: 20, step: 0.1 },
      min_cps: { min: 0, max: 10, step: 0.5 },
      max_cps: { min: 5, max: 60, step: 0.5 },
      max_gap_seconds: { min: 0.1, max: 5, step: 0.1 },
      lead_in_ms: { min: 0, max: 1000, step: 10 },
      tail_ms: { min: 0, max: 1000, step: 10 },
    },
  },
}

afterEach(() => {
  vi.restoreAllMocks()
  sessionStorage.clear()
  window.history.replaceState({}, '', '/')
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
    expect(await screen.findByText(/Manual quote and invoice before paid processing/i)).toBeVisible()
  })

  it('serves the dedicated payment and refund policy route', () => {
    window.history.replaceState({}, '', '/payments')
    render(<App />)

    expect(screen.getByRole('heading', { name: 'Payments and Refunds' })).toBeVisible()
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
    expect(JSON.parse(String((options?.body as FormData).get('style')))).toEqual({ source: 'preset', preset: 'standard' })
    expect(await screen.findByText('12 cues ready')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Download SRT' })).toBeEnabled()
  })

  it('offers preset and custom subtitle styles only for audio generation', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      id: 'styled-job', token: 'styled-token', mode: 'generate', status: 'complete', progress: 100,
      result: { cue_count: 3, cost_usd: 0.01 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
    }), { status: 202 }))
    const user = userEvent.setup()
    render(<App />)

    expect(screen.queryByRole('group', { name: 'Subtitle style source' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    expect(screen.getByRole('group', { name: 'Subtitle style source' })).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Custom' }))
    await user.clear(screen.getByLabelText('Characters per line'))
    await user.type(screen.getByLabelText('Characters per line'), '34')
    await user.clear(screen.getByLabelText('Maximum CPS'))
    await user.type(screen.getByLabelText('Maximum CPS'), '21')
    await user.upload(screen.getByLabelText('Dialogue audio'), new File(['audio'], 'episode.wav', { type: 'audio/wav' }))
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    const style = JSON.parse(String((fetchMock.mock.calls[1][1]?.body as FormData).get('style')))
    expect(style.source).toBe('custom')
    expect(style.values.max_chars_per_line).toBe(34)
    expect(style.values.max_cps).toBe(21)
  })

  it('requires and uploads an SRT example when sample-derived style is selected', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      id: 'sample-job', token: 'sample-token', mode: 'generate', status: 'complete', progress: 100,
      result: { cue_count: 2, cost_usd: 0.01 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
    }), { status: 202 }))
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    await user.click(screen.getByRole('button', { name: 'From SRT' }))
    await user.upload(screen.getByLabelText('Dialogue audio'), new File(['audio'], 'episode.wav', { type: 'audio/wav' }))
    expect(screen.getByRole('button', { name: 'Generate SRT' })).toBeDisabled()
    const example = new File(['1\n00:00:00,000 --> 00:00:01,000\nExample.\n'], 'example.srt', { type: 'application/x-subrip' })
    await user.upload(screen.getByLabelText('Style example SRT'), example)
    expect(screen.getByRole('button', { name: 'Generate SRT' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    const body = fetchMock.mock.calls[1][1]?.body as FormData
    expect(JSON.parse(String(body.get('style')))).toEqual({ source: 'sample' })
    expect(body.get('style_sample')).toBe(example)
  })

  it('requires the access code issued with a manual quote', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ...configResponse,
      access_code_required: true,
    }), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      id: 'quoted-job', token: 'job-token', mode: 'generate', status: 'complete', progress: 100,
      result: { cue_count: 1, cost_usd: 0.01 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
    }), { status: 202 }))
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    await user.upload(screen.getByLabelText('Dialogue audio'), new File(['audio'], 'episode.wav', { type: 'audio/wav' }))
    expect(screen.getByRole('button', { name: 'Generate SRT' })).toBeDisabled()
    await user.type(screen.getByLabelText('Job access code'), 'quote-code-1234')
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect((fetchMock.mock.calls[1][1]?.body as FormData).get('access_code')).toBe('quote-code-1234')
  })

  it('disables production intake when the access gate is not configured', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      ...configResponse,
      access_code_required: true,
      jobs_available: false,
    }), { status: 200 }))
    render(<App />)

    expect(await screen.findByRole('status')).toHaveTextContent('Job intake is temporarily unavailable')
    expect(screen.getByRole('button', { name: 'Start sync' })).toBeDisabled()
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
