import { AudioLines, FileText, LockKeyhole, Play } from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'

import { createJob, downloadJobArtifact, loadJob } from '../api'
import { clearActiveJob, readActiveJob, writeActiveJob } from '../session'
import type { JobMode, JobResponse, PublicConfig } from '../types'
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
  const [job, setJob] = useState<JobResponse | null>(null)
  const [token, setToken] = useState(restoredAccess?.token || '')
  const [submitting, setSubmitting] = useState(false)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [error, setError] = useState('')
  const canSubmit = useMemo(() => Boolean(audio && (mode === 'generate' || subtitle) && !submitting), [audio, mode, subtitle, submitting])

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
    if (!audio || (mode === 'sync' && !subtitle)) return
    setSubmitting(true)
    setError('')
    const body = new FormData()
    body.set('mode', mode)
    body.set('audio', audio)
    if (mode === 'sync' && subtitle) body.set('subtitle', subtitle)
    body.set('fps', fps)
    body.set('language', language)
    body.set('style', 'standard')
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
          <h1>Sync dialogue. Keep every cue honest.</h1>
          <p>Frame-accurate subtitle timing from the audio itself.</p>
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
          <div className="workspace-options">
            <label><span>Frame rate</span><select value={fps} onChange={(event) => setFps(event.target.value)}>{config.fps_values.map((value) => <option key={value} value={value}>{value} fps</option>)}</select></label>
            <label><span>Language</span><select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="auto">Auto-detect</option><option value="de">German</option><option value="fr">French</option><option value="en">English</option><option value="id">Indonesian</option><option value="es">Spanish</option></select></label>
            <label><span>Style</span><select value="standard" disabled><option value="standard">Standard</option></select></label>
            <button className="primary-button" type="submit" disabled={!canSubmit}><Play />{submitting ? 'Uploading' : mode === 'sync' ? 'Start sync' : 'Generate SRT'}</button>
          </div>
          {error && <div className="form-error" role="alert">{error}</div>}
        </form>
        <div className="retention-note"><LockKeyhole />Files are deleted after {config.retention_hours} hours</div>
        <WaveformPreview file={audio} />
        {job && <JobPanel job={job} onDownload={download} downloading={downloading} />}
      </div>
    </section>
  )
}

function messageFrom(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong.'
}
