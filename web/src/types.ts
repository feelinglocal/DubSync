export type JobMode = 'sync' | 'generate'
export type JobStatus = 'queued' | 'processing' | 'complete' | 'failed'

export interface PricingTier {
  usd_per_minute: number
  minimum_usd: number
}

export interface PublicConfig {
  retention_hours: number
  max_upload_bytes: number
  audio_extensions: string[]
  fps_values: number[]
  pricing: Record<'generate' | 'sync' | 'precision', PricingTier>
  billing_enabled: boolean
}

export interface JobResult {
  cue_count: number
  cost_usd: number
}

export interface JobResponse {
  id: string
  token?: string
  mode: JobMode
  status: JobStatus
  progress: number
  expires_at: string
  error: string | null
  result: JobResult | null
  downloads: string[]
}

export const defaultConfig: PublicConfig = {
  retention_hours: 24,
  max_upload_bytes: 2_147_483_648,
  audio_extensions: ['.aac', '.flac', '.m4a', '.mp3', '.ogg', '.wav'],
  fps_values: [23.976, 24, 25, 29.97, 30],
  pricing: {
    generate: { usd_per_minute: 0.12, minimum_usd: 3 },
    sync: { usd_per_minute: 0.18, minimum_usd: 5 },
    precision: { usd_per_minute: 0.25, minimum_usd: 10 },
  },
  billing_enabled: false,
}
