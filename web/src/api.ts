import type { JobResponse, PublicConfig } from './types'

export async function loadConfig(): Promise<PublicConfig> {
  const response = await fetch('/api/config')
  if (!response.ok) throw new Error('Could not load service configuration.')
  return response.json() as Promise<PublicConfig>
}

export async function createJob(body: FormData): Promise<JobResponse> {
  const response = await fetch('/api/jobs', { method: 'POST', body })
  const payload = (await response.json()) as JobResponse & { detail?: string }
  if (!response.ok) throw new Error(payload.detail || 'Could not start the job.')
  return payload
}

export async function loadJob(jobId: string, token: string): Promise<JobResponse> {
  const response = await fetch(`/api/jobs/${jobId}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error('Could not refresh the job.')
  return response.json() as Promise<JobResponse>
}

export async function downloadJobArtifact(jobId: string, token: string, kind: string): Promise<void> {
  const response = await fetch(`/api/jobs/${jobId}/downloads/${kind}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error('Could not download this file.')
  const blob = await response.blob()
  const disposition = response.headers.get('content-disposition') || ''
  const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] || `dubsync-${kind}`
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
