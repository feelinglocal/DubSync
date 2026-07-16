import { AudioLines, ChevronDown, FileText, LockKeyhole, Play } from 'lucide-react'
import { FormEvent, useEffect, useId, useMemo, useRef, useState } from 'react'

import { ApiError, createBatch, createJob, downloadJobArtifact, loadJob } from '../api'
import { pairSyncFiles, validateAudioFiles } from '../batch'
import {
  clearActiveJobs,
  readActiveJobs,
  writeActiveJobs,
  type ActiveJobAccess,
} from '../session'
import type { GenerationStyleValues, JobMode, JobResponse, PublicConfig } from '../types'
import { BatchUploadField } from './BatchUploadField'
import {
  GenerationStylePanel,
  type GenerationStyleDraft,
  validateGenerationStyleDraft,
} from './GenerationStylePanel'
import { JobPanel } from './JobPanel'
import { WaveformPreview } from './WaveformPreview'

interface DownloadState {
  jobId: string
  kind: string
}

export function Workspace({ config }: { config: PublicConfig }) {
  const batchNamingHelpId = useId()
  const [accesses, setAccesses] = useState<ActiveJobAccess[]>(readActiveJobs)
  const initialAccesses = useRef(accesses)
  const initiallyRestoringIds = useRef(new Set(accesses.map((access) => access.id)))
  const restoreRequest = useRef<Promise<PromiseSettledResult<JobResponse>[]> | null>(null)
  const [mode, setMode] = useState<JobMode>('sync')
  const [audioFiles, setAudioFiles] = useState<File[]>([])
  const [subtitleFiles, setSubtitleFiles] = useState<File[]>([])
  const [fps, setFps] = useState('30')
  const [language, setLanguage] = useState('auto')
  const [styleSource, setStyleSource] = useState<'preset' | 'custom' | 'sample'>('preset')
  const [stylePreset, setStylePreset] = useState(config.generation_styles.default_preset)
  const [customStyle, setCustomStyle] = useState<GenerationStyleDraft>(DEFAULT_STYLE_DRAFT)
  const [styleSample, setStyleSample] = useState<File | null>(null)
  const [accessCode, setAccessCode] = useState('')
  const [jobs, setJobs] = useState<JobResponse[]>([])
  const [pollRevision, setPollRevision] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [downloading, setDownloading] = useState<DownloadState | null>(null)
  const [error, setError] = useState('')
  const [refreshErrors, setRefreshErrors] = useState<Record<string, string>>({})
  const customStyleValues = useMemo(() => valuesFromDraft(customStyle), [customStyle])
  const customStyleValidation = useMemo(
    () => validateGenerationStyleDraft(customStyle, config.generation_styles),
    [config.generation_styles, customStyle],
  )
  const syncPairing = useMemo(
    () => pairSyncFiles(audioFiles, subtitleFiles),
    [audioFiles, subtitleFiles],
  )
  const audioValidation = useMemo(() => validateAudioFiles(audioFiles), [audioFiles])
  const selectionError = mode === 'sync' ? syncPairing.error : audioValidation
  const selectionTouched = audioFiles.length > 0 || subtitleFiles.length > 0
  const hasBlockingJob = useMemo(() => {
    const knownJobIds = new Set(jobs.map((job) => job.id))
    return jobs.some((job) => ['queued', 'processing'].includes(job.status))
      || accesses.some((access) => !knownJobIds.has(access.id))
  }, [accesses, jobs])
  const canSubmit = useMemo(
    () => Boolean(
      config.jobs_available
      && !selectionError
      && (mode !== 'generate' || styleSource !== 'sample' || styleSample)
      && (mode !== 'generate' || styleSource !== 'custom' || customStyleValidation.valid)
      && (!config.access_code_required || accessCode.trim())
      && !hasBlockingJob
      && !submitting
    ),
    [accessCode, config.access_code_required, config.jobs_available, customStyleValidation.valid, hasBlockingJob, mode, selectionError, styleSample, styleSource, submitting],
  )

  useEffect(() => {
    if (config.generation_styles.presets.some((preset) => preset.id === stylePreset)) return
    setStylePreset(config.generation_styles.default_preset)
  }, [config.generation_styles, stylePreset])

  useEffect(() => {
    persistAccesses(accesses)
  }, [accesses])

  useEffect(() => {
    const pendingAccesses = initialAccesses.current
    if (pendingAccesses.length === 0) return
    restoreRequest.current ??= Promise.allSettled(
      pendingAccesses.map((access) => loadJob(access.id, access.token)),
    )
    let active = true
    void restoreRequest.current.then((results) => {
      pendingAccesses.forEach((access) => initiallyRestoringIds.current.delete(access.id))
      if (!active) return
      const restoredJobs: JobResponse[] = []
      const retainedAccesses: ActiveJobAccess[] = []
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          restoredJobs.push(result.value)
          retainedAccesses.push(pendingAccesses[index])
        } else if (!isTerminalAccessError(result.reason)) {
          retainedAccesses.push(pendingAccesses[index])
        }
      })
      setJobs(restoredJobs)
      setAccesses(retainedAccesses)
      setRefreshErrors(Object.fromEntries(
        results.flatMap((result, index) => (
          result.status === 'rejected'
            ? [[
                pendingAccesses[index].id,
                refreshFailureMessage(pendingAccesses[index].id, jobs),
              ]]
            : []
        )),
      ))
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    const pollTimers: number[] = []
    const scheduledJobIds = new Set<string>()
    jobs.forEach((job) => {
      if (!['queued', 'processing'].includes(job.status)) return
      const access = accesses.find((candidate) => candidate.id === job.id)
      if (!access) return
      scheduledJobIds.add(job.id)
      const timer = window.setTimeout(() => {
        void loadJob(job.id, access.token)
          .then((updatedJob) => {
            clearRefreshError(job.id)
            setJobs((current) => current.map((candidate) => (
              candidate.id === updatedJob.id ? updatedJob : candidate
            )))
          })
          .catch((pollError) => handleRefreshFailure(job.id, pollError))
      }, 1500)
      pollTimers.push(timer)
    })

    accesses.forEach((access) => {
      if (
        initiallyRestoringIds.current.has(access.id)
        || scheduledJobIds.has(access.id)
        || jobs.some((job) => job.id === access.id)
      ) return
      const timer = window.setTimeout(() => {
        void loadJob(access.id, access.token)
          .then((restoredJob) => {
            clearRefreshError(access.id)
            setJobs((current) => orderJobsByAccess([...current, restoredJob], accesses))
          })
          .catch((restoreError) => handleRefreshFailure(access.id, restoreError))
      }, 1500)
      pollTimers.push(timer)
    })

    return () => pollTimers.forEach((timer) => window.clearTimeout(timer))
  }, [accesses, jobs, pollRevision])

  function handleRefreshFailure(jobId: string, refreshError: unknown) {
    setRefreshErrors((current) => ({
      ...current,
      [jobId]: refreshFailureMessage(jobId, jobs),
    }))
    if (isTerminalAccessError(refreshError)) {
      setAccesses((current) => current.filter((access) => access.id !== jobId))
      setJobs((current) => current.filter((job) => job.id !== jobId))
      return
    }
    setPollRevision((current) => current + 1)
  }

  function clearRefreshError(jobId: string) {
    setRefreshErrors((current) => Object.fromEntries(
      Object.entries(current).filter(([candidateId]) => candidateId !== jobId),
    ))
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)
    setError('')
    const body = new FormData()
    body.set('mode', mode)
    if (mode === 'sync') {
      syncPairing.pairs.forEach(({ audio, subtitle }) => {
        body.append('audio', audio)
        body.append('subtitle', subtitle)
      })
    } else {
      audioFiles.forEach((audio) => body.append('audio', audio))
    }
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
    const requestAccessCode = config.access_code_required ? accessCode.trim() : ''
    try {
      const createdJobs = audioFiles.length > 1
        ? (await createBatch(body, requestAccessCode)).jobs
        : [await createJob(body, requestAccessCode)]
      const nextAccesses = createdJobs.flatMap((job) => (
        job.token ? [{ id: job.id, token: job.token }] : []
      ))
      setJobs((current) => mergeById(current, createdJobs))
      setAccesses((current) => mergeById(current, nextAccesses))
    } catch (submitError) {
      setError(messageFrom(submitError))
    } finally {
      setSubmitting(false)
    }
  }

  async function download(job: JobResponse, kind: string) {
    const token = accesses.find((access) => access.id === job.id)?.token
    if (!token) return
    setDownloading({ jobId: job.id, kind })
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
    if (nextMode === mode) return
    setMode(nextMode)
    setError('')
    if (nextMode === 'generate') setSubtitleFiles([])
  }

  function selectAudio(files: File[]) {
    setAudioFiles(files)
    setError('')
  }

  function selectSubtitles(files: File[]) {
    setSubtitleFiles(files)
    setError('')
  }

  const isBatch = jobs.length > 1 || jobs.some((job) => Boolean(job.batch_id))

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
            <BatchUploadField label="Dialogue audio" kind="audio" accept={config.audio_extensions.join(',')} files={audioFiles} required describedBy={mode === 'sync' ? batchNamingHelpId : undefined} onChange={selectAudio} />
            {mode === 'sync' && <BatchUploadField label="Original SRT" kind="subtitle" accept=".srt,application/x-subrip,text/plain" files={subtitleFiles} maxBytes={config.max_srt_bytes} required describedBy={batchNamingHelpId} onChange={selectSubtitles} />}
          </div>
          {mode === 'sync' && <p className="batch-naming-help" id={batchNamingHelpId}>Match names: 001.wav + 001.srt. Up to 10 pairs.</p>}
          {mode === 'generate' && (
            <GenerationStylePanel
              config={config.generation_styles}
              source={styleSource}
              preset={stylePreset}
              values={customStyle}
              validation={customStyleValidation}
              sample={styleSample}
              maxSrtBytes={config.max_srt_bytes}
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
          {!config.jobs_available && <div className="service-notice" role="status">Job intake is temporarily unavailable. Contact <a href="mailto:rey@feelslocal.com">rey@feelslocal.com</a>.</div>}
          {selectionTouched && selectionError && <div className="form-error" role="alert">{selectionError}</div>}
          {error && <div className="form-error" role="alert">{error}</div>}
          {Object.entries(refreshErrors).map(([jobId, message]) => (
            <div className="form-error" role="alert" key={jobId}>{message}</div>
          ))}
        </form>
        <div className="retention-note"><LockKeyhole />Files are deleted after {config.retention_hours} hours</div>
        {audioFiles.length === 1 && <WaveformPreview file={audioFiles[0]} />}
        {isBatch ? (
          <section className="batch-results" aria-labelledby="batch-results-title">
            <h2 id="batch-results-title">Batch results</h2>
            <div className="batch-results-list">
              {jobs.map((job, index) => (
                <JobPanel
                  key={job.id}
                  job={job}
                  sourceName={job.source_name || `File ${index + 1}`}
                  onDownload={(kind) => download(job, kind)}
                  downloading={downloading?.jobId === job.id ? downloading.kind : null}
                />
              ))}
            </div>
          </section>
        ) : jobs[0] ? (
          <JobPanel
            job={jobs[0]}
            onDownload={(kind) => download(jobs[0], kind)}
            downloading={downloading?.jobId === jobs[0].id ? downloading.kind : null}
          />
        ) : null}
      </div>
    </section>
  )
}

function persistAccesses(accesses: readonly ActiveJobAccess[]) {
  if (accesses.length > 0) writeActiveJobs(accesses)
  else clearActiveJobs()
}

function orderJobsByAccess(jobs: readonly JobResponse[], accesses: readonly ActiveJobAccess[]) {
  const jobsById = new Map(jobs.map((job) => [job.id, job]))
  return accesses.flatMap((access) => {
    const job = jobsById.get(access.id)
    return job ? [job] : []
  })
}

function mergeById<T extends { id: string }>(current: readonly T[], incoming: readonly T[]): T[] {
  const merged = new Map(current.map((item) => [item.id, item]))
  incoming.forEach((item) => merged.set(item.id, item))
  return [...merged.values()]
}

function isTerminalAccessError(error: unknown) {
  return error instanceof ApiError && [401, 403, 404, 410].includes(error.status)
}

function refreshFailureMessage(jobId: string, jobs: readonly JobResponse[]) {
  const sourceName = jobs.find((job) => job.id === jobId)?.source_name
  return `Could not refresh job "${sourceName || jobId}".`
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
