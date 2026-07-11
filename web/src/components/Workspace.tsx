import { AudioLines, ChevronDown, FileText, LockKeyhole, Play } from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'

import { createJob, downloadJobArtifact, loadJob } from '../api'
import { clearActiveJob, readActiveJob, writeActiveJob } from '../session'
import type { GenerationStyleValues, JobMode, JobResponse, PublicConfig } from '../types'
import {
  GenerationStylePanel,
  type GenerationStyleDraft,
  validateGenerationStyleDraft,
} from './GenerationStylePanel'
import { JobPanel } from './JobPanel'
import { UploadField } from './UploadField'
import { WaveformPreview } from './WaveformPreview'

export function Workspace({ config }: { config: PublicConfig }) {
  const [restoredAccess, setRestoredAccess] = useState(readActiveJob)
  const [mode, setMode] = useState<JobMode>('sync')
  const [audio, setAudio] = useState<File | null>(null)
  const [subtitle, setSubtitle] = useState<File | null>(null)
  const [fps, setFps] = useState('30')
  const [language, setLanguage] = useState('auto')
  const [styleSource, setStyleSource] = useState<'preset' | 'custom' | 'sample'>('preset')
  const [stylePreset, setStylePreset] = useState(config.generation_styles.default_preset)
  const [customStyle, setCustomStyle] = useState<GenerationStyleDraft>(DEFAULT_STYLE_DRAFT)
  const [styleSample, setStyleSample] = useState<File | null>(null)
  const [accessCode, setAccessCode] = useState('')
  const [job, setJob] = useState<JobResponse | null>(null)
  const [token, setToken] = useState(restoredAccess?.token || '')
  const [submitting, setSubmitting] = useState(false)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [error, setError] = useState('')
  const customStyleValues = useMemo(() => valuesFromDraft(customStyle), [customStyle])
  const customStyleValidation = useMemo(
    () => validateGenerationStyleDraft(customStyle, config.generation_styles),
    [config.generation_styles, customStyle],
  )
  const canSubmit = useMemo(
    () => Boolean(
      config.jobs_available
      && audio
      && (mode === 'generate' || subtitle)
      && (mode !== 'generate' || styleSource !== 'sample' || styleSample)
      && (mode !== 'generate' || styleSource !== 'custom' || customStyleValidation.valid)
      && (!config.access_code_required || accessCode.trim())
      && !submitting
    ),
    [accessCode, audio, config.access_code_required, config.jobs_available, customStyleValidation.valid, mode, styleSample, styleSource, subtitle, submitting],
  )

  useEffect(() => {
    if (config.generation_styles.presets.some((preset) => preset.id === stylePreset)) return
    setStylePreset(config.generation_styles.default_preset)
  }, [config.generation_styles, stylePreset])

  useEffect(() => {
    if (!restoredAccess || job) return
    loadJob(restoredAccess.id, restoredAccess.token)
      .then(setJob)
      .catch((restoreError) => {
        clearActiveJob()
        setToken('')
        setError(messageFrom(restoreError))
      })
  }, [job, restoredAccess])

  useEffect(() => {
    if (!job || !token || !['queued', 'processing'].includes(job.status)) return
    const timeout = window.setTimeout(async () => {
      try {
        setJob(await loadJob(job.id, token))
      } catch (pollError) {
        setError(messageFrom(pollError))
      }
    }, 1500)
    return () => window.clearTimeout(timeout)
  }, [job, token])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (
      !audio
      || (mode === 'sync' && !subtitle)
      || (mode === 'generate' && styleSource === 'sample' && !styleSample)
      || (mode === 'generate' && styleSource === 'custom' && !customStyleValidation.valid)
    ) return
    setSubmitting(true)
    setError('')
    const body = new FormData()
    body.set('mode', mode)
    body.set('audio', audio)
    if (mode === 'sync' && subtitle) body.set('subtitle', subtitle)
    body.set('fps', fps)
    body.set('language', language)
    if (mode === 'generate') {
      const style = styleSource === 'preset'
        ? { source: 'preset', preset: stylePreset }
        : styleSource === 'custom'
          ? { source: 'custom', values: customStyleValues }
          : { source: 'sample' }
      body.set('style', JSON.stringify(style))
      if (styleSource === 'sample' && styleSample) body.set('style_sample', styleSample)
    } else {
      body.set('style', 'source')
    }
    if (config.access_code_required) body.set('access_code', accessCode.trim())
    try {
      const created = await createJob(body)
      const nextToken = created.token || ''
      setToken(nextToken)
      setJob(created)
      if (nextToken) writeActiveJob({ id: created.id, token: nextToken })
    } catch (submitError) {
      setError(messageFrom(submitError))
    } finally {
      setSubmitting(false)
    }
  }

  async function download(kind: string) {
    if (!job || !token) return
    setDownloading(kind)
    setError('')
    try {
      await downloadJobArtifact(job.id, token, kind)
    } catch (downloadError) {
      setError(messageFrom(downloadError))
    } finally {
      setDownloading(null)
    }
  }

  function selectMode(nextMode: JobMode) {
    setMode(nextMode)
    setJob(null)
    setToken('')
    setRestoredAccess(null)
    clearActiveJob()
    setError('')
    if (nextMode === 'generate') setSubtitle(null)
  }

  return (
    <section className="workspace-section" id="workspace">
      <div className="workspace-inner">
        <div className="workspace-heading">
          <h1>SRT sync that follows the performance.</h1>
          <p>Sync existing subtitles to dubbed audio, or generate speaker-aware captions from audio.</p>
        </div>
        <form onSubmit={submit} className="workspace-form" noValidate>
          <div className="mode-control" role="group" aria-label="Workflow mode">
            <button type="button" className={mode === 'sync' ? 'is-selected' : ''} onClick={() => selectMode('sync')} aria-pressed={mode === 'sync'}><AudioLines />Sync existing SRT</button>
            <button type="button" className={mode === 'generate' ? 'is-selected' : ''} onClick={() => selectMode('generate')} aria-pressed={mode === 'generate'}><FileText />Generate from audio</button>
          </div>
          <div className="upload-stack">
            <UploadField label="Dialogue audio" kind="audio" accept={config.audio_extensions.join(',')} file={audio} required onChange={setAudio} />
            {mode === 'sync' && <UploadField label="Original SRT" kind="subtitle" accept=".srt,application/x-subrip,text/plain" file={subtitle} required onChange={setSubtitle} />}
          </div>
          {mode === 'generate' && (
            <GenerationStylePanel
              config={config.generation_styles}
              source={styleSource}
              preset={stylePreset}
              values={customStyle}
              validation={customStyleValidation}
              sample={styleSample}
              onSourceChange={setStyleSource}
              onPresetChange={setStylePreset}
              onValuesChange={setCustomStyle}
              onSampleChange={setStyleSample}
            />
          )}
          <div className={config.access_code_required ? 'workspace-options has-access-code' : 'workspace-options'}>
            <label><span className="field-label">Frame rate</span><span className="select-control"><select value={fps} onChange={(event) => setFps(event.target.value)}>{config.fps_values.map((value) => <option key={value} value={value}>{value} fps</option>)}</select><ChevronDown aria-hidden="true" /></span></label>
            <label><span className="field-label">Language</span><span className="select-control"><select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="auto">Auto-detect</option><option value="de">German</option><option value="fr">French</option><option value="en">English</option><option value="id">Indonesian</option><option value="es">Spanish</option></select><ChevronDown aria-hidden="true" /></span></label>
            {config.access_code_required && config.jobs_available && (
              <label><span className="field-label">Job access code</span><input type="password" value={accessCode} onChange={(event) => setAccessCode(event.target.value)} autoComplete="one-time-code" required /></label>
            )}
            <button className="primary-button" type="submit" disabled={!canSubmit}><Play />{submitting ? 'Uploading' : mode === 'sync' ? 'Start sync' : 'Generate SRT'}</button>
          </div>
          {!config.jobs_available && <div className="service-notice" role="status">Job intake is temporarily unavailable. Contact <a href="mailto:reyhanputraph@gmail.com">reyhanputraph@gmail.com</a>.</div>}
          {error && <div className="form-error" role="alert">{error}</div>}
        </form>
        <div className="retention-note"><LockKeyhole />Files are deleted after {config.retention_hours} hours</div>
        {audio && <WaveformPreview file={audio} />}
        {job && <JobPanel job={job} onDownload={download} downloading={downloading} />}
      </div>
    </section>
  )
}

function messageFrom(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong.'
}

const DEFAULT_STYLE_DRAFT: GenerationStyleDraft = {
  max_lines_per_cue: '2',
  max_chars_per_line: '26',
  min_cue_duration_seconds: '0.5',
  max_cue_duration_seconds: '5',
  min_cps: '2',
  max_cps: '30',
  max_gap_seconds: '0.8',
  lead_in_ms: '0',
  tail_ms: '40',
}

function valuesFromDraft(draft: GenerationStyleDraft): GenerationStyleValues {
  return {
    max_lines_per_cue: draftNumber(draft.max_lines_per_cue),
    max_chars_per_line: draftNumber(draft.max_chars_per_line),
    min_cue_duration_seconds: draftNumber(draft.min_cue_duration_seconds),
    max_cue_duration_seconds: draftNumber(draft.max_cue_duration_seconds),
    min_cps: draftNumber(draft.min_cps),
    max_cps: draftNumber(draft.max_cps),
    max_gap_seconds: draftNumber(draft.max_gap_seconds),
    lead_in_ms: draftNumber(draft.lead_in_ms),
    tail_ms: draftNumber(draft.tail_ms),
  }
}

function draftNumber(value: string) {
  return value.trim() === '' ? Number.NaN : Number(value)
}
