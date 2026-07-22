import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'

const configResponse = {
  retention_hours: 24,
  max_upload_bytes: 536_870_912,
  max_srt_bytes: 2_097_152,
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

const completedBatchResponse = {
  id: 'batch-1',
  jobs: [
    {
      id: 'job-1', token: 'token-1', source_name: '001', batch_id: 'batch-1', batch_position: 0,
      mode: 'sync', status: 'complete', progress: 100,
      result: { cue_count: 3, cost_usd: 0.01 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
    },
    {
      id: 'job-2', token: 'token-2', source_name: '002', batch_id: 'batch-1', batch_position: 1,
      mode: 'sync', status: 'complete', progress: 100,
      result: { cue_count: 4, cost_usd: 0.02 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
    },
  ],
}

afterEach(() => {
  vi.restoreAllMocks()
  sessionStorage.clear()
  window.history.replaceState({}, '', '/')
  document.title = ''
  document.head.querySelector('link[rel="canonical"]')?.remove()
  document.head.querySelector('meta[name="description"]')?.remove()
  document.head.querySelector('meta[property="og:title"]')?.remove()
})

describe('DubSync workspace', () => {
  it('shows the real sync workflow and requires both source files by default', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(configResponse), { status: 200 }))
    render(<App />)

    expect(screen.getByRole('heading', { name: 'SRT sync that follows the performance.' })).toBeVisible()
    expect(screen.getByLabelText('Dialogue audio')).toBeRequired()
    expect(screen.getByLabelText('Original SRT')).toBeRequired()
    expect(screen.getByRole('button', { name: 'Start sync' })).toBeDisabled()
    expect(await screen.findByText('Files are deleted after 24 hours')).toBeVisible()
    expect(await screen.findByText(/Manual quote and invoice before paid processing/i)).toBeVisible()
    expect(document.title).toBe('Subtitle Sync & Audio-to-SRT for Dubbing | DubSync')
    expect(document.head.querySelector('link[rel="canonical"]')).toHaveAttribute('href', 'https://dubsync.onrender.com/')
    expect(document.head.querySelector('meta[name="description"]')).toHaveAttribute('content', expect.stringContaining('Sync an existing SRT'))
  })

  it('shows the simple file-pair naming instruction for batch sync', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(configResponse), { status: 200 }))

    render(<App />)

    const help = 'Match names: 001.wav + 001.srt. Up to 10 pairs.'
    expect(await screen.findByText(help)).toBeVisible()
    expect(screen.getByLabelText('Dialogue audio')).toHaveAccessibleDescription(help)
    expect(screen.getByLabelText('Original SRT')).toHaveAccessibleDescription(help)
  })

  it('shows the configured SRT limit for original and style-example uploads', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(configResponse), { status: 200 }))
    const user = userEvent.setup()

    render(<App />)

    expect(await screen.findByText('SRT up to 2 MB each')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    await user.click(screen.getByRole('button', { name: 'From SRT' }))
    expect(screen.getByText('SRT up to 2 MB')).toBeVisible()
  })

  it('keeps batch submission disabled when audio and subtitle stems do not match', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(configResponse), { status: 200 }))
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await user.upload(screen.getByLabelText('Dialogue audio'), [
      new File(['audio-1'], '001.wav', { type: 'audio/wav' }),
      new File(['audio-2'], '002.wav', { type: 'audio/wav' }),
    ])
    await user.upload(screen.getByLabelText('Original SRT'), [
      new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' }),
      new File(['subtitle-3'], '003.srt', { type: 'application/x-subrip' }),
    ])

    expect(screen.getByRole('button', { name: 'Start sync' })).toBeDisabled()
  })

  it('submits two named pairs in one multipart batch request and renders both child results', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(completedBatchResponse), { status: 202 }))
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
    const firstAudio = new File(['audio-1'], '001.wav', { type: 'audio/wav' })
    const secondAudio = new File(['audio-2'], '002.wav', { type: 'audio/wav' })
    const firstSubtitle = new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' })
    const secondSubtitle = new File(['subtitle-2'], '002.srt', { type: 'application/x-subrip' })

    await user.upload(screen.getByLabelText('Dialogue audio'), [firstAudio, secondAudio])
    await user.upload(screen.getByLabelText('Original SRT'), [secondSubtitle, firstSubtitle])
    await user.click(screen.getByRole('button', { name: 'Start sync' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    const [url, options] = fetchMock.mock.calls[1]
    const body = options?.body as FormData
    expect(url).toBe('/api/batches')
    expect(options?.method).toBe('POST')
    expect((body.getAll('audio') as File[]).map((file) => file.name)).toEqual(['001.wav', '002.wav'])
    expect((body.getAll('subtitle') as File[]).map((file) => file.name)).toEqual(['001.srt', '002.srt'])
    expect(await screen.findByText('001')).toBeVisible()
    expect(screen.getByText('3 cues ready')).toBeVisible()
    expect(screen.getByText('002')).toBeVisible()
    expect(screen.getByText('4 cues ready')).toBeVisible()
  })

  it('uses the selected child token when downloading a batch result', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(completedBatchResponse), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(new Blob(['subtitle']), {
      status: 200,
      headers: { 'content-disposition': 'attachment; filename="002-dubsync-synced.srt"' },
    }))
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await user.upload(screen.getByLabelText('Dialogue audio'), [
      new File(['audio-1'], '001.wav', { type: 'audio/wav' }),
      new File(['audio-2'], '002.wav', { type: 'audio/wav' }),
    ])
    await user.upload(screen.getByLabelText('Original SRT'), [
      new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' }),
      new File(['subtitle-2'], '002.srt', { type: 'application/x-subrip' }),
    ])
    await user.click(screen.getByRole('button', { name: 'Start sync' }))
    await user.click(await screen.findByRole('button', { name: 'Download 002 SRT' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    expect(fetchMock.mock.calls[2][0]).toBe('/api/jobs/job-2/downloads/srt')
    expect(fetchMock.mock.calls[2][1]?.headers).toEqual({ Authorization: 'Bearer token-2' })
  })

  it('places one ZIP download action in the completed batch header and sends every child token', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(completedBatchResponse), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(new Blob(['zip']), {
      status: 200,
      headers: { 'content-disposition': 'attachment; filename="dubsync-batch-batch-1-synced-srts.zip"' },
    }))
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await user.upload(screen.getByLabelText('Dialogue audio'), [
      new File(['audio-1'], '001.wav', { type: 'audio/wav' }),
      new File(['audio-2'], '002.wav', { type: 'audio/wav' }),
    ])
    await user.upload(screen.getByLabelText('Original SRT'), [
      new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' }),
      new File(['subtitle-2'], '002.srt', { type: 'application/x-subrip' }),
    ])
    await user.click(screen.getByRole('button', { name: 'Start sync' }))

    const downloadAll = await screen.findByRole('button', { name: 'Download all SRTs' })
    expect(downloadAll.closest('.batch-results-header')).not.toBeNull()
    await user.click(downloadAll)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    expect(fetchMock.mock.calls[2][0]).toBe('/api/batches/batch-1/downloads/srt')
    expect(fetchMock.mock.calls[2][1]).toMatchObject({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    })
    expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
      jobs: [
        { id: 'job-1', token: 'token-1' },
        { id: 'job-2', token: 'token-2' },
      ],
    })
  })

  it('shows a completed-SRT ZIP action after a partial batch finishes', async () => {
    const partialBatch = {
      ...completedBatchResponse,
      jobs: completedBatchResponse.jobs.map((job, index) => (
        index === 0
          ? job
          : { ...job, status: 'failed', progress: 100, result: null, downloads: [], error: 'Job failed.' }
      )),
    }
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(partialBatch), { status: 202 }))
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await user.upload(screen.getByLabelText('Dialogue audio'), [
      new File(['audio-1'], '001.wav', { type: 'audio/wav' }),
      new File(['audio-2'], '002.wav', { type: 'audio/wav' }),
    ])
    await user.upload(screen.getByLabelText('Original SRT'), [
      new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' }),
      new File(['subtitle-2'], '002.srt', { type: 'application/x-subrip' }),
    ])
    await user.click(screen.getByRole('button', { name: 'Start sync' }))

    expect(await screen.findByRole('button', { name: 'Download completed SRTs' })).toBeVisible()
  })

  it('keeps every child token through mode switches and later completed submissions', async () => {
    const laterJob = {
      id: 'job-3', token: 'token-3', source_name: 'episode',
      mode: 'generate', status: 'complete', progress: 100,
      result: { cue_count: 8, cost_usd: 0.03 }, downloads: ['srt'],
      expires_at: '2026-07-12T00:00:00Z', error: null,
    }
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(configResponse), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(completedBatchResponse), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(laterJob), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(new Blob(['subtitle']), {
      status: 200,
      headers: { 'content-disposition': 'attachment; filename="002-dubsync-synced.srt"' },
    }))
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await user.upload(screen.getByLabelText('Dialogue audio'), [
      new File(['audio-1'], '001.wav', { type: 'audio/wav' }),
      new File(['audio-2'], '002.wav', { type: 'audio/wav' }),
    ])
    await user.upload(screen.getByLabelText('Original SRT'), [
      new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' }),
      new File(['subtitle-2'], '002.srt', { type: 'application/x-subrip' }),
    ])
    await user.click(screen.getByRole('button', { name: 'Start sync' }))
    expect(await screen.findByText('4 cues ready')).toBeVisible()
    await waitFor(() => expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual([
      { id: 'job-1', token: 'token-1' },
      { id: 'job-2', token: 'token-2' },
    ]))

    await user.click(screen.getByRole('button', { name: 'Sync existing SRT' }))
    expect(screen.getByText('4 cues ready')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    expect(screen.getByText('3 cues ready')).toBeVisible()
    expect(screen.getByText('4 cues ready')).toBeVisible()

    await user.upload(
      screen.getByLabelText('Dialogue audio'),
      new File(['audio-3'], 'episode.wav', { type: 'audio/wav' }),
    )
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))

    expect(await screen.findByText('8 cues ready')).toBeVisible()
    expect(screen.getByText('3 cues ready')).toBeVisible()
    expect(screen.getByText('4 cues ready')).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Download all SRTs' })).not.toBeInTheDocument()
    await waitFor(() => expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual([
      { id: 'job-1', token: 'token-1' },
      { id: 'job-2', token: 'token-2' },
      { id: 'job-3', token: 'token-3' },
    ]))

    await user.click(screen.getByRole('button', { name: 'Download 002 SRT' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4))
    expect(fetchMock.mock.calls[3][0]).toBe('/api/jobs/job-2/downloads/srt')
    expect(fetchMock.mock.calls[3][1]?.headers).toEqual({ Authorization: 'Bearer token-2' })
  })

  it('prevents another submission while any child job is queued or processing', async () => {
    const queuedBatch = {
      id: 'batch-queued',
      jobs: completedBatchResponse.jobs.map((job, index) => ({
        ...job,
        id: `queued-${index + 1}`,
        token: `queued-token-${index + 1}`,
        status: index === 0 ? 'queued' : 'processing',
        progress: index === 0 ? 0 : 25,
        result: null,
        downloads: [],
      })),
    }
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, options) => {
      if (String(input) === '/api/config') {
        return new Response(JSON.stringify(configResponse), { status: 200 })
      }
      if (options?.method === 'POST') {
        return new Response(JSON.stringify(queuedBatch), { status: 202 })
      }
      const jobId = String(input).split('/').at(-1)
      const job = queuedBatch.jobs.find((candidate) => candidate.id === jobId)
      return new Response(JSON.stringify(job), { status: job ? 200 : 404 })
    })
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await user.upload(screen.getByLabelText('Dialogue audio'), [
      new File(['audio-1'], '001.wav', { type: 'audio/wav' }),
      new File(['audio-2'], '002.wav', { type: 'audio/wav' }),
    ])
    await user.upload(screen.getByLabelText('Original SRT'), [
      new File(['subtitle-1'], '001.srt', { type: 'application/x-subrip' }),
      new File(['subtitle-2'], '002.srt', { type: 'application/x-subrip' }),
    ])
    await user.click(screen.getByRole('button', { name: 'Start sync' }))

    expect(await screen.findByText('Waiting to start')).toBeVisible()
    expect(screen.getByText('Processing dialogue')).toBeVisible()
    expect(screen.getByRole('progressbar', { name: '001 progress' })).toHaveAttribute('value', '0')
    expect(screen.getByRole('progressbar', { name: '002 progress' })).toHaveAttribute('value', '25')
    expect(screen.getByRole('button', { name: 'Start sync' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Download all SRTs' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate from audio' }))
    await user.upload(
      screen.getByLabelText('Dialogue audio'),
      new File(['audio-3'], 'episode.wav', { type: 'audio/wav' }),
    )
    expect(screen.getByRole('button', { name: 'Generate SRT' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Generate SRT' }))

    expect(fetchMock.mock.calls.filter(([, options]) => options?.method === 'POST')).toHaveLength(1)
    expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual([
      { id: 'queued-1', token: 'queued-token-1' },
      { id: 'queued-2', token: 'queued-token-2' },
    ])
  })

  it('serves the dedicated payment and refund policy route', () => {
    window.history.replaceState({}, '', '/payments')
    render(<App />)

    expect(screen.getByRole('heading', { name: 'Payments and Refunds' })).toBeVisible()
    expect(document.title).toBe('Payments and Refunds | DubSync')
    expect(document.head.querySelector('link[rel="canonical"]')).toHaveAttribute('href', 'https://dubsync.onrender.com/payments')
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
    await user.clear(screen.getByLabelText('Minimum CPS'))
    expect(screen.getByRole('button', { name: 'Generate SRT' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('Minimum CPS is required.')
    await user.type(screen.getByLabelText('Minimum CPS'), '3')
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
    expect(fetchMock.mock.calls[1][1]?.headers).toEqual({
      'X-DubSync-Access-Code': 'quote-code-1234',
    })
    expect((fetchMock.mock.calls[1][1]?.body as FormData).has('access_code')).toBe(false)
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
    sessionStorage.clear()
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
    sessionStorage.clear()
    sessionStorage.setItem('dubsync:active-job', JSON.stringify({ id: 'expired-job', token: 'expired-token' }))
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      if (String(input) === '/api/config') return new Response(JSON.stringify(configResponse), { status: 200 })
      return new Response('', { status: 404 })
    })

    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not refresh job "expired-job".')
    expect(sessionStorage.getItem('dubsync:active-job')).toBeNull()
    expect(sessionStorage.getItem('dubsync:active-jobs')).toBeNull()
  })

  it('keeps restored access after a transient refresh failure', async () => {
    const access = { id: 'retry-job', token: 'retry-token' }
    sessionStorage.setItem('dubsync:active-jobs', JSON.stringify([access]))
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      if (String(input) === '/api/config') return new Response(JSON.stringify(configResponse), { status: 200 })
      return new Response('', { status: 503 })
    })

    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not refresh job "retry-job".')
    expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual([access])
  })

  it('retries polling a queued job after a transient failure', async () => {
    vi.useFakeTimers()
    const access = { id: 'poll-job', token: 'poll-token' }
    sessionStorage.setItem('dubsync:active-jobs', JSON.stringify([access]))
    let jobLoads = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      if (String(input) === '/api/config') return new Response(JSON.stringify(configResponse), { status: 200 })
      jobLoads += 1
      if (jobLoads === 1) {
        return new Response(JSON.stringify({
          id: 'poll-job', mode: 'sync', status: 'queued', progress: 0,
          result: null, downloads: [], expires_at: '2026-07-12T00:00:00Z', error: null,
        }), { status: 200 })
      }
      if (jobLoads === 2) return new Response('', { status: 503 })
      return new Response(JSON.stringify({
        id: 'poll-job', mode: 'sync', status: 'complete', progress: 100,
        result: { cue_count: 9, cost_usd: 0.02 }, downloads: ['srt'], expires_at: '2026-07-12T00:00:00Z', error: null,
      }), { status: 200 })
    })

    try {
      render(<App />)
      await flushReactUpdates()
      expect(screen.getByText('Waiting to start')).toBeVisible()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500)
      })
      expect(jobLoads).toBe(2)
      expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual([access])

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500)
      })
      expect(jobLoads).toBe(3)
      expect(screen.getByText('9 cues ready')).toBeVisible()
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not clear one child refresh error when a different child recovers', async () => {
    vi.useFakeTimers()
    sessionStorage.setItem('dubsync:active-jobs', JSON.stringify([
      { id: 'expired-child', token: 'expired-token' },
      { id: 'recovering-child', token: 'recovering-token' },
    ]))
    let recoveringLoads = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/config') return new Response(JSON.stringify(configResponse), { status: 200 })
      if (url === '/api/jobs/expired-child') return new Response('', { status: 404 })
      if (url === '/api/jobs/recovering-child') {
        recoveringLoads += 1
        const complete = recoveringLoads > 1
        return new Response(JSON.stringify({
          id: 'recovering-child', mode: 'sync', status: complete ? 'complete' : 'queued',
          progress: complete ? 100 : 0,
          result: complete ? { cue_count: 4, cost_usd: 0.01 } : null,
          downloads: complete ? ['srt'] : [], expires_at: '2026-07-12T00:00:00Z', error: null,
        }), { status: 200 })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    try {
      render(<App />)
      await flushReactUpdates()
      expect(screen.getByRole('alert')).toHaveTextContent('Could not refresh job "expired-child".')

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500)
      })

      expect(screen.getByText('4 cues ready')).toBeVisible()
      expect(screen.getByRole('alert')).toHaveTextContent('Could not refresh job "expired-child".')
    } finally {
      vi.useRealTimers()
    }
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

async function flushReactUpdates() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}
