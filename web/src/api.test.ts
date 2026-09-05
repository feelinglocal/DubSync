import { afterEach, describe, expect, it, vi } from 'vitest'

import { createBatch, createJob, downloadBatchSrtArchive, downloadJobArtifact, loadConfig, loadJob } from './api'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('web API client', () => {
  it('loads public configuration and reports a failed request', async () => {
    const payload = { retention_hours: 12 }
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(payload), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response('', { status: 503 }))

    await expect(loadConfig()).resolves.toEqual(payload)
    await expect(loadConfig()).rejects.toThrow('Could not load service configuration.')
  })

  it('creates jobs and preserves useful API validation errors', async () => {
    const body = new FormData()
    const created = { id: 'job-1', token: 'token-1' }
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(created), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Audio is empty.' }), { status: 422 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({}), { status: 500 }))

    await expect(createJob(body)).resolves.toEqual(created)
    await expect(createJob(body)).rejects.toThrow('Audio is empty.')
    await expect(createJob(body)).rejects.toThrow('Could not start the job.')
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/jobs', { method: 'POST', body })
  })

  it('sends the quote access code in a preflight header for jobs and batches', async () => {
    const body = new FormData()
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ id: 'job-1' }), { status: 202 }))
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ id: 'batch-1', jobs: [] }), { status: 202 }))

    await createJob(body, 'quote-code-1234')
    await createBatch(body, 'quote-code-1234')

    const expectedOptions = {
      method: 'POST',
      body,
      headers: { 'X-DubSync-Access-Code': 'quote-code-1234' },
    }
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/jobs', expectedOptions)
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/batches', expectedOptions)
    expect(body.has('access_code')).toBe(false)
  })

  it('shows actionable upload failures when a proxy returns HTML instead of JSON', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response('<html>Bad gateway</html>', { status: 502 }))
    await expect(createJob(new FormData())).rejects.toThrow('Could not start the job. The service returned an unreadable response (HTTP 502).')
    await expect(createBatch(new FormData())).rejects.toThrow('Could not start the batch. The service returned an unreadable response (HTTP 502).')
  })

  it('loads a protected job and rejects an unavailable one', async () => {
    const job = { id: 'job-2', status: 'processing' }
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(job), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response('', { status: 404 }))

    await expect(loadJob('job-2', 'secret')).resolves.toEqual(job)
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/jobs/job-2', {
      headers: { Authorization: 'Bearer secret' },
      signal: expect.any(AbortSignal),
    })
    await expect(loadJob('missing', 'secret')).rejects.toThrow('Could not refresh the job.')
  })

  it('aborts a stalled status refresh so recovery can retry', async () => {
    vi.useFakeTimers()
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((_input, options) => (
      new Promise<Response>((_resolve, reject) => {
        options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), { once: true })
      })
    ))
    try {
      const request = loadJob('stalled-job', 'secret')
      const outcome = expect(request).rejects.toThrow('Job refresh timed out. Retrying automatically.')
      await vi.advanceTimersByTimeAsync(30_000)
      expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
      await outcome
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('downloads a protected artifact with the server filename', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:download')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Blob(['subtitle']), {
        status: 200,
        headers: { 'content-disposition': 'attachment; filename="episode.synced.srt"' },
      }),
    )

    await downloadJobArtifact('job-3', 'secret', 'srt')

    expect(click).toHaveBeenCalledOnce()
    expect(createUrl).toHaveBeenCalledOnce()
    expect(revokeUrl).toHaveBeenCalledWith('blob:download')
  })

  it('downloads every protected batch SRT as one server-named ZIP', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.download).toBe('dubsync-batch-batch-1-synced-srts.zip')
    })
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:batch-download')
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Blob(['zip']), {
        status: 200,
        headers: { 'content-disposition': 'attachment; filename="dubsync-batch-batch-1-synced-srts.zip"' },
      }),
    )
    const jobs = [
      { id: 'job-1', token: 'token-1' },
      { id: 'job-2', token: 'token-2' },
    ]

    await downloadBatchSrtArchive('batch-1', jobs)

    expect(fetchMock).toHaveBeenCalledWith('/api/batches/batch-1/downloads/srt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobs }),
    })
    expect(click).toHaveBeenCalledOnce()
  })

  it('uses a fallback filename and reports failed downloads', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.download).toBe('dubsync-qc-json')
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(new Blob(['{}']), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response('', { status: 404 }))

    await downloadJobArtifact('job-4', 'secret', 'qc-json')
    expect(click).toHaveBeenCalledOnce()
    await expect(downloadJobArtifact('job-4', 'secret', 'srt')).rejects.toThrow('Could not download this file.')
  })

  it('decodes RFC 5987 download filenames used for spaces and Unicode', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.download).toBe('Caf\u00e9 episode-dubsync-synced.srt')
    })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(new Blob(['subtitle']), {
        status: 200,
        headers: {
          'content-disposition': "attachment; filename*=utf-8''Caf%C3%A9%20episode-dubsync-synced.srt",
        },
      }),
    )

    await downloadJobArtifact('job-5', 'secret', 'srt')

    expect(click).toHaveBeenCalledOnce()
  })
})
