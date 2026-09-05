import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { JobResponse } from '../types'
import { Header } from './Header'
import { JobPanel } from './JobPanel'
import { LegalPage } from './LegalPage'
import { UploadField } from './UploadField'
import { downsampleWaveform, formatAudioDuration, WaveformPreview } from './WaveformPreview'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

function useSystemTheme(theme: 'light' | 'dark') {
  const listeners = new Set<(event: MediaQueryListEvent) => void>()
  vi.stubGlobal('matchMedia', vi.fn().mockImplementation((query: string) => ({
    matches: query === '(prefers-color-scheme: dark)' && theme === 'dark',
    media: query,
    onchange: null,
    addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })))
}

function job(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    id: 'job-1',
    mode: 'generate',
    status: 'queued',
    progress: 10,
    expires_at: '2026-07-12T00:00:00Z',
    error: null,
    result: null,
    downloads: [],
    ...overrides,
  }
}

describe('shared components', () => {
  it('opens and closes the compact navigation', async () => {
    useSystemTheme('light')
    const user = userEvent.setup()
    render(<Header />)
    await user.click(screen.getByRole('button', { name: 'Open menu' }))
    expect(screen.getByRole('button', { name: 'Close menu' })).toHaveAttribute('aria-expanded', 'true')
    await user.click(screen.getByRole('link', { name: 'Subtitle QC' }))
    expect(screen.getByRole('button', { name: 'Open menu' })).toHaveAttribute('aria-expanded', 'false')
  })

  it('uses the system theme until the visitor chooses and persists an override', async () => {
    useSystemTheme('dark')
    const user = userEvent.setup()
    const { container } = render(<Header />)

    expect(document.documentElement).toHaveAttribute('data-theme', 'dark')
    expect(container.querySelector('img.brand-mark')).toHaveAttribute('src', '/brand/dubsync-mark.svg')

    const toggle = screen.getByRole('button', { name: 'Use light theme' })
    await user.click(toggle)

    expect(document.documentElement).toHaveAttribute('data-theme', 'light')
    expect(localStorage.getItem('dubsync-theme')).toBe('light')
    expect(screen.getByRole('button', { name: 'Use dark theme' })).toBeVisible()
  })

  it('restores a saved theme instead of overriding it with the system theme', () => {
    useSystemTheme('dark')
    localStorage.setItem('dubsync-theme', 'light')

    render(<Header />)

    expect(document.documentElement).toHaveAttribute('data-theme', 'light')
    expect(screen.getByRole('button', { name: 'Use dark theme' })).toBeVisible()
  })

  it('renders the terms, privacy, and payment policies', () => {
    const { rerender } = render(<LegalPage kind="terms" />)
    expect(screen.getByRole('heading', { name: 'Terms of Service' })).toBeVisible()
    expect(screen.getAllByText(/Reyhan Putra/).length).toBeGreaterThan(0)
    rerender(<LegalPage kind="privacy" />)
    expect(screen.getByRole('heading', { name: 'Privacy Policy' })).toBeVisible()
    expect(screen.getByRole('heading', { name: /Service providers/ })).toBeVisible()
    expect(screen.getByText(/Google Gemini for audio-aware adjudication and bounded punctuation/)).toBeVisible()
    expect(screen.getByText(/OpenAI for speaker mapping/)).toBeVisible()
    rerender(<LegalPage kind="payments" />)
    expect(screen.getByRole('heading', { name: 'Payments and Refunds' })).toBeVisible()
    expect(screen.getByRole('heading', { name: /When a full refund applies/ })).toBeVisible()
    expect(screen.getByText(/within 7 calendar days/i)).toBeVisible()
    expect(screen.getByText(/applicable taxes/i)).toBeVisible()
  })

  it('links the payment policy from the legal navigation', () => {
    render(<LegalPage kind="terms" />)
    expect(screen.getByRole('link', { name: 'Payments' })).toHaveAttribute('href', '/payments')
  })

  it('renders queued, processing, failed, and complete job states', async () => {
    const onDownload = vi.fn()
    const { rerender } = render(<JobPanel job={job()} onDownload={onDownload} downloading={null} />)
    expect(screen.getByText('Waiting to start')).toBeVisible()
    expect(screen.getByRole('progressbar')).toHaveValue(10)

    rerender(<JobPanel job={job({ status: 'processing', progress: 55 })} onDownload={onDownload} downloading={null} />)
    expect(screen.getByText('Processing dialogue')).toBeVisible()

    rerender(<JobPanel job={job({ status: 'failed', error: 'Provider unavailable.' })} onDownload={onDownload} downloading={null} />)
    expect(screen.getByText('Job failed')).toBeVisible()
    expect(screen.getByText('Provider unavailable.')).toHaveClass('job-message', 'is-error')
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()

    rerender(<JobPanel job={job({ status: 'complete', progress: 100, result: { cue_count: 18, cost_usd: 0.03, fps: 24, fps_source: 'detected', fps_detection_confident: true }, downloads: ['srt', 'qc-json', 'qc-html'] })} onDownload={onDownload} downloading={null} />)
    expect(screen.getByText('18 cues ready')).toBeVisible()
    expect(screen.getByText('Your result and QC files are ready.')).toHaveClass('job-message')
    expect(screen.getByText('24 fps detected')).toHaveClass('job-detail')
    await userEvent.click(screen.getByRole('button', { name: 'QC JSON' }))
    await userEvent.click(screen.getByRole('button', { name: 'QC report' }))
    await userEvent.click(screen.getByRole('button', { name: 'Download SRT' }))
    expect(onDownload.mock.calls.map(([kind]) => kind)).toEqual(['qc-json', 'qc-html', 'srt'])

    rerender(<JobPanel job={job({ status: 'complete', progress: 100, result: { cue_count: 1, cost_usd: 0 }, downloads: ['srt'] })} onDownload={onDownload} downloading="srt" />)
    expect(screen.getByRole('button', { name: 'Download SRT' })).toBeDisabled()
  })

  it('selects, formats, and removes an uploaded file', async () => {
    function Harness() {
      const [file, setFile] = useState<File | null>(null)
      return <UploadField label="Dialogue audio" accept=".wav" file={file} kind="audio" onChange={setFile} />
    }
    const user = userEvent.setup()
    render(<Harness />)
    expect(screen.getByText('Choose dialogue audio')).toBeVisible()
    await user.upload(screen.getByLabelText('Dialogue audio'), new File([new Uint8Array(1024 * 1024)], 'episode.wav', { type: 'audio/wav' }))
    expect(screen.getByText('episode.wav')).toBeVisible()
    expect(screen.getByText('1.0 MB')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Remove dialogue audio' }))
    expect(screen.getByText('Choose dialogue audio')).toBeVisible()
  })

  it('downsamples real decoded audio peaks without inventing cue content', () => {
    expect(downsampleWaveform(new Float32Array([0, 0.5, -1, 0.25]), 2)).toEqual([0.5, 1])
    expect(downsampleWaveform(new Float32Array(), 3)).toEqual([0, 0, 0])
    expect(formatAudioDuration(59.6)).toBe('1:00 audio')
    expect(formatAudioDuration(Number.NaN)).toBe('Waveform ready')
  })

  it('draws an audio preview and replaces the object URL when the file changes', async () => {
    const context = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      strokeStyle: '',
      lineWidth: 0,
    }
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D)
    const createUrl = vi.spyOn(URL, 'createObjectURL')
      .mockReturnValueOnce('blob:first')
      .mockReturnValueOnce('blob:second')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const first = new File(['one'], 'first.wav', { type: 'audio/wav' })
    const second = new File(['two'], 'second.wav', { type: 'audio/wav' })

    const { rerender, unmount } = render(<WaveformPreview file={first} />)
    expect(await screen.findByLabelText('Audio preview player')).toHaveAttribute('src', 'blob:first')
    expect(context.stroke).toHaveBeenCalled()
    expect(screen.queryByLabelText('Subtitle cue preview')).not.toBeInTheDocument()
    expect(screen.queryByText('Every word has a place.')).not.toBeInTheDocument()

    rerender(<WaveformPreview file={second} />)
    await waitFor(() => expect(screen.getByLabelText('Audio preview player')).toHaveAttribute('src', 'blob:second'))
    expect(revokeUrl).toHaveBeenCalledWith('blob:first')
    expect(createUrl).toHaveBeenCalledTimes(2)

    unmount()
    expect(revokeUrl).toHaveBeenCalledWith('blob:second')
  })

  it('decodes the selected audio and draws peaks from its real channel data', async () => {
    const context = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      strokeStyle: '',
      lineWidth: 0,
    }
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D)
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:decoded')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    vi.spyOn(File.prototype, 'arrayBuffer').mockResolvedValue(new ArrayBuffer(16))
    const close = vi.fn().mockResolvedValue(undefined)
    const decodeAudioData = vi.fn().mockResolvedValue({
      duration: 65.2,
      getChannelData: () => new Float32Array([0, 0.2, -0.8, 0.4, -1, 0.1]),
    })
    class FakeAudioContext {
      decodeAudioData = decodeAudioData
      close = close
    }
    vi.stubGlobal('AudioContext', FakeAudioContext)

    render(<WaveformPreview file={new File(['audio'], 'decoded.wav', { type: 'audio/wav' })} />)

    const player = screen.getByLabelText('Audio preview player')
    expect(decodeAudioData).not.toHaveBeenCalled()
    Object.defineProperty(player, 'duration', { value: 65.2 })
    fireEvent.loadedMetadata(player)

    expect(await screen.findByText('1:05 audio')).toBeVisible()
    expect(decodeAudioData).toHaveBeenCalledOnce()
    expect(context.lineTo.mock.calls.length).toBeGreaterThan(2)
    await waitFor(() => expect(close).toHaveBeenCalled())
  })

  it('makes QC warnings visible and distinguishes informational notes from review findings', () => {
    const completed = job({ status: 'complete', result: {
      cue_count: 18, cost_usd: 0,
      qc_summary: { flags: 4, style_violations: 2, error_count: 1, warning_count: 3, info_count: 2 },
    }, downloads: ['srt', 'qc-html'] })
    const { rerender } = render(<JobPanel job={completed} onDownload={vi.fn()} downloading={null} />)
    expect(screen.getByText('18 cues processed · QC review needed')).toBeVisible()
    expect(screen.getByText('1 QC error · 3 QC warnings. Review the QC report before using this SRT.')).toBeVisible()
    expect(screen.getByRole('button', { name: 'QC report' })).toBeEnabled()

    rerender(<JobPanel job={{ ...completed, result: {
      cue_count: 18, cost_usd: 0,
      qc_summary: { flags: 2, style_violations: 0, error_count: 0, warning_count: 0, info_count: 2 },
    } }} onDownload={vi.fn()} downloading={null} />)
    expect(screen.getByText('18 cues ready')).toBeVisible()
    expect(screen.getByText('No automated QC warnings or errors · 2 informational notes.')).toBeVisible()
  })

  it('shows unavailable or legacy QC summaries without claiming an automated clean result', () => {
    const completed = job({ status: 'complete', result: { cue_count: 1, cost_usd: 0 }, downloads: ['srt', 'qc-html'] })
    const { rerender } = render(<JobPanel job={completed} onDownload={vi.fn()} downloading={null} />)
    expect(screen.getByText('QC summary unavailable. Review the QC report.')).toBeVisible()
    rerender(<JobPanel job={{ ...completed, result: {
      cue_count: 1, cost_usd: 0, qc_summary: { flags: 2, style_violations: 1 },
    } }} onDownload={vi.fn()} downloading={null} />)
    expect(screen.getByText('2 QC flags · 1 style issue. Review the QC report.')).toBeVisible()
  })

  it('keeps long audio playable without reading or decoding the whole recording for its waveform', async () => {
    const arrayBuffer = vi.spyOn(File.prototype, 'arrayBuffer').mockResolvedValue(new ArrayBuffer(16))
    const decodeAudioData = vi.fn().mockResolvedValue({
      duration: 2700,
      getChannelData: () => new Float32Array([0, 0.5]),
    })
    class FakeAudioContext {
      decodeAudioData = decodeAudioData
      close = vi.fn().mockResolvedValue(undefined)
    }
    vi.stubGlobal('AudioContext', FakeAudioContext)
    render(<WaveformPreview file={new File(['compressed audio'], 'episode.mp3', { type: 'audio/mpeg' })} />)
    const player = screen.getByLabelText('Audio preview player')
    Object.defineProperty(player, 'duration', { value: 2700 })
    fireEvent.loadedMetadata(player)

    expect(await screen.findByText(/Waveform preview is limited to audio up to 5 minutes and 16 MB/)).toBeVisible()
    expect(player).toHaveAttribute('controls')
    expect(arrayBuffer).not.toHaveBeenCalled()
    expect(decodeAudioData).not.toHaveBeenCalled()
  })

  it('skips oversized waveform files before allocating an audio context', async () => {
    const file = new File(['audio'], 'large.wav', { type: 'audio/wav' })
    Object.defineProperty(file, 'size', { value: 17 * 1024 * 1024 })
    const arrayBuffer = vi.spyOn(File.prototype, 'arrayBuffer').mockResolvedValue(new ArrayBuffer(16))
    const createContext = vi.fn()
    class FakeAudioContext {
      constructor() { createContext() }
      decodeAudioData = vi.fn().mockRejectedValue(new Error('Should not decode'))
      close = vi.fn().mockResolvedValue(undefined)
    }
    vi.stubGlobal('AudioContext', FakeAudioContext)
    render(<WaveformPreview file={file} />)

    expect(await screen.findByText(/Waveform preview is limited to audio up to 5 minutes and 16 MB/)).toBeVisible()
    expect(screen.getByLabelText('Audio preview player')).toHaveAttribute('controls')
    expect(createContext).not.toHaveBeenCalled()
    expect(arrayBuffer).not.toHaveBeenCalled()
  })
})
