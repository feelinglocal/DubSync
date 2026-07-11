import { afterEach, describe, expect, it, vi } from 'vitest'

import { createJob, downloadJobArtifact, loadConfig, loadJob } from './api'

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

  it('loads a protected job and rejects an unavailable one', async () => {
    const job = { id: 'job-2', status: 'processing' }
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(job), { status: 200 }))
    fetchMock.mockResolvedValueOnce(new Response('', { status: 404 }))

    await expect(loadJob('job-2', 'secret')).resolves.toEqual(job)
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/jobs/job-2', {
      headers: { Authorization: 'Bearer secret' },
    })
    await expect(loadJob('missing', 'secret')).rejects.toThrow('Could not refresh the job.')
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
})

