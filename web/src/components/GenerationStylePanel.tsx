import { ChevronDown } from 'lucide-react'
import { useId } from 'react'

import type {
  GenerationStyleSource,
  GenerationStylesConfig,
  GenerationStyleValueKey,
} from '../types'
import { UploadField } from './UploadField'

export type GenerationStyleDraft = Record<GenerationStyleValueKey, string>

export interface GenerationStyleDraftValidation {
  valid: boolean
  message: string
  invalidFields: GenerationStyleValueKey[]
}

interface GenerationStylePanelProps {
  config: GenerationStylesConfig
  source: GenerationStyleSource
  preset: string
  values: GenerationStyleDraft
  validation: GenerationStyleDraftValidation
  sample: File | null
  maxSrtBytes: number
  onSourceChange: (source: GenerationStyleSource) => void
  onPresetChange: (preset: string) => void
  onValuesChange: (values: GenerationStyleDraft) => void
  onSampleChange: (file: File | null) => void
}

const CUSTOM_FIELDS: ReadonlyArray<{ key: GenerationStyleValueKey; label: string }> = [
  { key: 'max_lines_per_cue', label: 'Lines per cue' },
  { key: 'max_chars_per_line', label: 'Characters per line' },
  { key: 'min_cue_duration_seconds', label: 'Minimum cue duration (s)' },
  { key: 'max_cue_duration_seconds', label: 'Maximum cue duration (s)' },
  { key: 'min_cps', label: 'Minimum CPS' },
  { key: 'max_cps', label: 'Maximum CPS' },
  { key: 'max_gap_seconds', label: 'Maximum speech gap (s)' },
  { key: 'lead_in_ms', label: 'Lead-in (ms)' },
  { key: 'tail_ms', label: 'Tail (ms)' },
]

export function GenerationStylePanel({
  config,
  source,
  preset,
  values,
  validation,
  sample,
  maxSrtBytes,
  onSourceChange,
  onPresetChange,
  onValuesChange,
  onSampleChange,
}: GenerationStylePanelProps) {
  const headingId = useId()
  const validationId = useId()
  return (
    <section className="subtitle-style-panel" aria-labelledby={headingId}>
      <div className="subtitle-style-heading">
        <h2 id={headingId}>Subtitle style</h2>
        <div className="style-source-control" role="group" aria-label="Subtitle style source">
          <button type="button" className={source === 'preset' ? 'is-selected' : ''} aria-pressed={source === 'preset'} onClick={() => onSourceChange('preset')}>Preset</button>
          <button type="button" className={source === 'custom' ? 'is-selected' : ''} aria-pressed={source === 'custom'} onClick={() => onSourceChange('custom')}>Custom</button>
          <button type="button" className={source === 'sample' ? 'is-selected' : ''} aria-pressed={source === 'sample'} onClick={() => onSourceChange('sample')}>From SRT</button>
        </div>
      </div>

      {source === 'preset' && (
        <label className="style-preset-field">
          <span className="field-label">Style preset</span>
          <span className="select-control">
            <select value={preset} onChange={(event) => onPresetChange(event.target.value)}>
              {config.presets.map((option) => <option key={option.id} value={option.id}>{option.name}</option>)}
            </select>
            <ChevronDown aria-hidden="true" />
          </span>
        </label>
      )}

      {source === 'custom' && (
        <div className="style-custom-grid" aria-describedby={validation.message ? validationId : undefined}>
          {CUSTOM_FIELDS.map((field) => {
            const limit = config.custom_limits[field.key]
            return (
              <label key={field.key}>
                <span className="field-label">{field.label}</span>
                <input
                  type="number"
                  inputMode="decimal"
                  min={limit.min}
                  max={limit.max}
                  step={limit.step}
                  value={values[field.key]}
                  aria-invalid={validation.invalidFields.includes(field.key) || undefined}
                  aria-describedby={validation.invalidFields.includes(field.key) ? validationId : undefined}
                  onChange={(event) => onValuesChange({ ...values, [field.key]: event.target.value })}
                />
              </label>
            )
          })}
        </div>
      )}
      {source === 'custom' && validation.message && (
        <p className="style-validation-message" id={validationId} role="alert">{validation.message}</p>
      )}

      {source === 'sample' && (
        <div className="style-sample-field">
          <UploadField
            label="Style example SRT"
            kind="style"
            accept=".srt,application/x-subrip,text/plain"
            file={sample}
            maxBytes={maxSrtBytes}
            required
            onChange={onSampleChange}
          />
        </div>
      )}
    </section>
  )
}

export function validateGenerationStyleDraft(
  draft: GenerationStyleDraft,
  config: GenerationStylesConfig,
): GenerationStyleDraftValidation {
  const emptyFields = CUSTOM_FIELDS.filter(({ key }) => draft[key].trim() === '').map(({ key }) => key)
  if (emptyFields.length) {
    return validationFailure(emptyFields, `${labelFor(emptyFields[0])} is required.`)
  }

  const outOfRangeFields = CUSTOM_FIELDS.filter(({ key }) => {
    const value = Number(draft[key])
    const limit = config.custom_limits[key]
    return !Number.isFinite(value) || value < limit.min || value > limit.max
  }).map(({ key }) => key)
  if (outOfRangeFields.length) {
    const key = outOfRangeFields[0]
    const limit = config.custom_limits[key]
    return validationFailure(outOfRangeFields, `${labelFor(key)} must be between ${limit.min} and ${limit.max}.`)
  }

  const values = numericValues(draft)
  if (values.min_cue_duration_seconds > values.max_cue_duration_seconds) {
    return validationFailure(
      ['min_cue_duration_seconds', 'max_cue_duration_seconds'],
      'Minimum cue duration cannot exceed maximum cue duration.',
    )
  }
  if (values.min_cps > values.max_cps) {
    return validationFailure(['min_cps', 'max_cps'], 'Minimum CPS cannot exceed maximum CPS.')
  }
  return { valid: true, message: '', invalidFields: [] }
}

function numericValues(draft: GenerationStyleDraft) {
  return Object.fromEntries(
    (Object.keys(draft) as GenerationStyleValueKey[]).map((key) => [key, Number(draft[key])]),
  ) as Record<GenerationStyleValueKey, number>
}

function validationFailure(
  invalidFields: GenerationStyleValueKey[],
  message: string,
): GenerationStyleDraftValidation {
  return { valid: false, message, invalidFields: [...invalidFields] }
}

function labelFor(key: GenerationStyleValueKey) {
  return CUSTOM_FIELDS.find((field) => field.key === key)?.label || 'Custom style value'
}
