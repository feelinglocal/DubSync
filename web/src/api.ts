import type { BatchResponse, JobResponse, PublicConfig } from './types'

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

export async function loadConfig(): Promise<PublicConfig> {
  const response = await fetch('/api/config')
  if (!response.ok) throw new Error('Could not load service configuration.')
  return response.json() as Promise<PublicConfig>
}

export async function createJob(body: FormData, accessCode = ''): Promise<JobResponse> {
  const response = await fetch('/api/jobs', createRequestOptions(body, accessCode))
  const payload = (await response.json()) as JobResponse & { detail?: string }
  if (!response.ok) throw new Error(payload.detail || 'Could not start the job.')
  return payload
}

export async function createBatch(body: FormData, accessCode = ''): Promise<BatchResponse> {
  const response = await fetch('/api/batches', createRequestOptions(body, accessCode))
  const payload = (await response.json()) as BatchResponse & { detail?: string }
  if (!response.ok) throw new Error(payload.detail || 'Could not start the batch.')
  return payload
}

export async function loadJob(jobId: string, token: string): Promise<JobResponse> {
  const response = await fetch(`/api/jobs/${jobId}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new ApiError('Could not refresh the job.', response.status)
  return response.json() as Promise<JobResponse>
}

export async function downloadJobArtifact(jobId: string, token: string, kind: string): Promise<void> {
  const response = await fetch(`/api/jobs/${jobId}/downloads/${kind}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error('Could not download this file.')
  await downloadResponse(response, `dubsync-${kind}`)
}

export async function downloadBatchSrtArchive(
  batchId: string,
  jobs: readonly { id: string; token: string }[],
): Promise<void> {
  const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}/downloads/srt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jobs }),
  })
  if (!response.ok) throw new Error('Could not download the completed SRTs.')
  await downloadResponse(response, `dubsync-batch-${batchId}-synced-srts.zip`)
}

async function downloadResponse(response: Response, fallbackFilename: string): Promise<void> {
  const blob = await response.blob()
  const disposition = response.headers.get('content-disposition') || ''
  const filename = filenameFromDisposition(disposition) || fallbackFilename
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function filenameFromDisposition(disposition: string): string | null {
  const encoded = disposition.match(/filename\*\s*=\s*utf-8'[^']*'([^;]+)/i)?.[1]
  if (encoded) {
    try {
      return decodeURIComponent(stripHeaderQuotes(encoded.trim()))
    } catch {
      // Fall through to the plain filename parameter when encoding is malformed.
    }
  }
  const plain = disposition.match(/filename\s*=\s*(?:"([^"]+)"|([^;]+))/i)
  return plain ? stripHeaderQuotes((plain[1] || plain[2]).trim()) : null
}

function stripHeaderQuotes(value: string): string {
  return value.replace(/^"|"$/g, '')
}

function createRequestOptions(body: FormData, accessCode: string): RequestInit {
  const headers = accessCode
    ? { 'X-DubSync-Access-Code': accessCode }
    : undefined
  return headers ? { method: 'POST', body, headers } : { method: 'POST', body }
}
