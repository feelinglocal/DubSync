export type JobMode = 'sync' | 'generate'
export type JobStatus = 'queued' | 'processing' | 'complete' | 'failed'

export interface PricingTier {
  usd_per_minute: number
  minimum_usd: number
}

export type GenerationStyleSource = 'preset' | 'custom' | 'sample'

export interface GenerationStyleValues {
  max_lines_per_cue: number
  max_chars_per_line: number
  min_cue_duration_seconds: number
  max_cue_duration_seconds: number
  min_cps: number
  max_cps: number
  max_gap_seconds: number
  lead_in_ms: number
  tail_ms: number
}

export type GenerationStyleValueKey = keyof GenerationStyleValues

export interface GenerationStyleLimit {
  min: number
  max: number
  step: number
}

export interface GenerationStylePreset {
  id: string
  name: string
  values: GenerationStyleValues
}

export interface GenerationStylesConfig {
  default_preset: string
  presets: GenerationStylePreset[]
  custom_limits: Record<GenerationStyleValueKey, GenerationStyleLimit>
}

export interface PublicConfig {
  retention_hours: number
  max_upload_bytes: number
  max_srt_bytes: number
  audio_extensions: string[]
  fps_values: number[]
  pricing: Record<'generate' | 'sync' | 'precision', PricingTier>
  billing_enabled: boolean
  access_code_required: boolean
  jobs_available: boolean
  generation_styles: GenerationStylesConfig
}

export interface JobResult {
  cue_count: number
  cost_usd: number
}

export interface JobResponse {
  id: string
  token?: string
  source_name?: string | null
  batch_id?: string | null
  batch_position?: number | null
  mode: JobMode
  status: JobStatus
  progress: number
  expires_at: string
  error: string | null
  result: JobResult | null
  downloads: string[]
}

export interface BatchResponse {
  id: string
  jobs: JobResponse[]
}

export const defaultConfig: PublicConfig = {
  retention_hours: 24,
  max_upload_bytes: 536_870_912,
  max_srt_bytes: 2_097_152,
  audio_extensions: ['.aac', '.flac', '.m4a', '.mp3', '.ogg', '.wav'],
  fps_values: [23.976, 24, 25, 29.97, 30],
  pricing: {
    generate: { usd_per_minute: 0.12, minimum_usd: 3 },
    sync: { usd_per_minute: 0.18, minimum_usd: 5 },
    precision: { usd_per_minute: 0.25, minimum_usd: 10 },
  },
  billing_enabled: false,
  access_code_required: false,
  jobs_available: false,
  generation_styles: {
    default_preset: 'standard',
    presets: [
      {
        id: 'standard',
        name: 'DubSync default',
        values: {
          max_lines_per_cue: 2,
          max_chars_per_line: 26,
          min_cue_duration_seconds: 0.5,
          max_cue_duration_seconds: 5,
          min_cps: 2,
          max_cps: 30,
          max_gap_seconds: 0.8,
          lead_in_ms: 0,
          tail_ms: 40,
        },
      },
      {
        id: 'streaming',
        name: 'Streaming',
        values: {
          max_lines_per_cue: 2,
          max_chars_per_line: 42,
          min_cue_duration_seconds: 1,
          max_cue_duration_seconds: 7,
          min_cps: 2,
          max_cps: 20,
          max_gap_seconds: 1,
          lead_in_ms: 0,
          tail_ms: 120,
        },
      },
      {
        id: 'broadcast',
        name: 'Broadcast',
        values: {
          max_lines_per_cue: 2,
          max_chars_per_line: 37,
          min_cue_duration_seconds: 1,
          max_cue_duration_seconds: 6,
          min_cps: 2,
          max_cps: 18,
          max_gap_seconds: 0.6,
          lead_in_ms: 0,
          tail_ms: 80,
        },
      },
      {
        id: 'short_form',
        name: 'Short-form',
        values: {
          max_lines_per_cue: 2,
          max_chars_per_line: 24,
          min_cue_duration_seconds: 0.4,
          max_cue_duration_seconds: 3.5,
          min_cps: 2,
          max_cps: 24,
          max_gap_seconds: 0.5,
          lead_in_ms: 0,
          tail_ms: 60,
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
