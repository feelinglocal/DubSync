import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { JobResponse } from '../types'
import { Header } from './Header'
import { JobPanel } from './JobPanel'
import { LegalPage } from './LegalPage'
import { UploadField } from './UploadField'
import { WaveformPreview } from './WaveformPreview'

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
    const user = userEvent.setup()
    render(<Header />)
    await user.click(screen.getByRole('button', { name: 'Open menu' }))
    expect(screen.getByRole('button', { name: 'Close menu' })).toHaveAttribute('aria-expanded', 'true')
    await user.click(screen.getByRole('button', { name: 'Close menu' }))
    expect(screen.getByRole('button', { name: 'Open menu' })).toHaveAttribute('aria-expanded', 'false')
  })

  it('renders both legal documents', () => {
    const { rerender } = render(<LegalPage kind="terms" />)
    expect(screen.getByRole('heading', { name: 'Terms of Service' })).toBeVisible()
    expect(screen.getByRole('heading', { name: /Fees and refunds/ })).toBeVisible()
    rerender(<LegalPage kind="privacy" />)
    expect(screen.getByRole('heading', { name: 'Privacy Policy' })).toBeVisible()
    expect(screen.getByRole('heading', { name: /Service providers/ })).toBeVisible()
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
    expect(screen.getByText('Provider unavailable.')).toBeVisible()
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()

    rerender(<JobPanel job={job({ status: 'complete', progress: 100, result: { cue_count: 18, cost_usd: 0.03 }, downloads: ['srt', 'qc-json', 'qc-html'] })} onDownload={onDownload} downloading={null} />)
    expect(screen.getByText('18 cues ready')).toBeVisible()
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

  it('draws a waveform and replaces the object URL when the audio changes', async () => {
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

    rerender(<WaveformPreview file={second} />)
    await waitFor(() => expect(screen.getByLabelText('Audio preview player')).toHaveAttribute('src', 'blob:second'))
    expect(revokeUrl).toHaveBeenCalledWith('blob:first')
    expect(createUrl).toHaveBeenCalledTimes(2)

    fireEvent.click(screen.getByRole('button', { name: 'Previous cue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next cue' }))
    unmount()
    expect(revokeUrl).toHaveBeenCalledWith('blob:second')
  })
})
