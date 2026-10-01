import { CheckCircle2, CircleHelp, Download, FileDiff, FileJson, FileText, LoaderCircle, TriangleAlert } from 'lucide-react'
import { useId } from 'react'

import { transcriptionModelLabels, type JobResponse, type QcVerdict } from '../types'

interface JobPanelProps {
  job: JobResponse
  onDownload: (kind: string) => void
  downloading: string | null
  sourceName?: string
}

type QcTone = 'ok' | 'warning' | 'error' | 'unknown'

interface QcStatus {
  tone: QcTone
  headline: string
  text: string
}

export function JobPanel({ job, onDownload, downloading, sourceName }: JobPanelProps) {
  const titleId = useId()
  const complete = job.status === 'complete'
  const failed = job.status === 'failed'
  const fpsStatus = complete ? formatFpsStatus(job.result) : null
  const qcStatus = complete ? formatQcStatus(job.result) : null
  const tone: QcTone = failed ? 'error' : qcStatus?.tone ?? 'ok'
  const transcriptionLabel = job.transcription_provider
    ? transcriptionModelLabels[job.transcription_provider === 'default' ? 'scribe_v2' : job.transcription_provider]
    : null
  return (
    <section className="job-panel" aria-live="polite" aria-labelledby={titleId}>
      <div className="job-summary">
        <span className={statusIconClass(tone, complete || failed)} aria-hidden="true">
          {!complete && !failed ? <LoaderCircle className="spin" />
            : tone === 'error' || tone === 'warning' ? <TriangleAlert />
              : tone === 'unknown' ? <CircleHelp /> : <CheckCircle2 />}
        </span>
        <div>
          <span className="section-label" id={sourceName ? undefined : titleId}>{sourceName ? 'Source' : 'Latest job'}</span>
          {sourceName && <span className="job-source-name" id={titleId}>{sourceName}</span>}
          <strong>{complete ? `${job.result?.cue_count ?? 0} cues ${qcStatus?.headline ?? 'ready'}` : failed ? 'Job failed' : job.status === 'processing' ? 'Processing dialogue' : 'Waiting to start'}</strong>
          <span className={failed ? 'job-message is-error' : 'job-message'}>{job.error || (complete ? 'Your result and QC files are ready.' : 'You can keep this page open while DubSync works.')}</span>
          {transcriptionLabel && <span className="job-detail">Transcription: {transcriptionLabel}</span>}
          {fpsStatus && <span className="job-detail">{fpsStatus}</span>}
          {qcStatus && <span className={qcMessageClass(qcStatus.tone)}>{qcStatus.text}</span>}
        </div>
        <span className="job-progress">{job.progress}%</span>
      </div>
      {!complete && !failed && <progress value={job.progress} max="100" aria-label={`${sourceName || 'Job'} progress`}>{job.progress}%</progress>}
      {complete && (
        <div className="download-actions">
          <button type="button" className="secondary-button" onClick={() => onDownload('srt')} disabled={downloading !== null} aria-label={sourceName ? `Download ${sourceName} SRT` : undefined}>
            <Download /> Download SRT
          </button>
          {job.downloads.includes('qc-json') && (
            <button type="button" className="icon-command" onClick={() => onDownload('qc-json')} disabled={downloading !== null} title="Download QC JSON" aria-label={sourceName ? `Download ${sourceName} QC JSON` : undefined}>
              <FileJson /><span>QC JSON</span>
            </button>
          )}
          {job.downloads.includes('qc-html') && (
            <button type="button" className="icon-command" onClick={() => onDownload('qc-html')} disabled={downloading !== null} title="Download QC report" aria-label={sourceName ? `Download ${sourceName} QC report` : undefined}>
              <FileText /><span>QC report</span>
            </button>
          )}
          {job.downloads.includes('changes') && (
            <button type="button" className="icon-command" onClick={() => onDownload('changes')} disabled={downloading !== null} title="Download change log (SRT)" aria-label={sourceName ? `Download ${sourceName} change log` : undefined}>
              <FileDiff /><span>Changes</span>
            </button>
          )}
        </div>
      )}
    </section>
  )
}

function statusIconClass(tone: QcTone, finished: boolean): string {
  if (!finished) return 'status-icon'
  if (tone === 'error') return 'status-icon is-error'
  if (tone === 'warning') return 'status-icon is-warning'
  if (tone === 'unknown') return 'status-icon is-neutral'
  return 'status-icon'
}

function qcMessageClass(tone: QcTone): string {
  if (tone === 'error') return 'job-message is-error'
  if (tone === 'warning') return 'job-message is-warning'
  return 'job-detail'
}

function formatQcStatus(result: JobResponse['result']): QcStatus {
  const summary = result?.qc_summary
  if (!summary || !isCount(summary.flags) || !isCount(summary.style_violations)) {
    // Never claim a clean result without a summary to back it.
    return { tone: 'unknown', headline: 'ready', text: 'QC summary unavailable. Review the QC report.' }
  }
  const verdict = formatVerdict(summary)
  if (verdict) return verdict
  // Reports written before the review tiers: only raw flag counts exist.
  const errors = summary.error_count
  const warnings = summary.warning_count
  const info = summary.info_count
  if (isCount(errors) && isCount(warnings) && isCount(info)
    && errors + warnings + info === summary.flags + summary.style_violations) {
    const findings = [
      errors > 0 ? countLabel(errors, 'QC error') : '',
      warnings > 0 ? countLabel(warnings, 'QC warning') : '',
    ].filter(Boolean)
    if (findings.length > 0) {
      return { tone: 'error', headline: 'processed · QC review needed', text: `${findings.join(' · ')}. Review the QC report before using this SRT.` }
    }
    return { tone: 'ok', headline: 'ready', text: `No automated QC warnings or errors${info > 0 ? ` · ${countLabel(info, 'informational note')}` : ''}.` }
  }
  const legacyFindings = summary.flags + summary.style_violations > 0
  return {
    tone: legacyFindings ? 'error' : 'ok',
    headline: legacyFindings ? 'processed · QC review needed' : 'ready',
    text: `${countLabel(summary.flags, 'QC flag')} · ${countLabel(summary.style_violations, 'style issue')}. Review the QC report.`,
  }
}

function formatVerdict(summary: NonNullable<NonNullable<JobResponse['result']>['qc_summary']>): QcStatus | null {
  const verdict: QcVerdict | undefined = summary.verdict
  const items = summary.review_item_count
  const errors = summary.review_error_count
  const warnings = summary.review_warning_count
  if (!verdict || !isCount(items) || !isCount(errors) || !isCount(warnings) || errors + warnings !== items) return null
  const changes = isCount(summary.change_count) ? ` · ${countLabel(summary.change_count, 'change')} logged` : ''
  if (verdict === 'clean' && items === 0) {
    return { tone: 'ok', headline: 'ready', text: `Nothing needs review${changes}.` }
  }
  if (verdict === 'check' && errors === 0) {
    const cues = isCount(summary.review_cue_count) ? ` (${countLabel(summary.review_cue_count, 'cue')})` : ''
    return {
      tone: 'warning',
      headline: `processed · ${countLabel(items, 'item')} to check`,
      text: `${countLabel(items, 'item')} to check${cues}${changes}. See the QC report.`,
    }
  }
  if (verdict === 'attention') {
    const parts = errors > 0
      ? [`${countLabel(errors, 'item')} need${errors === 1 ? 's' : ''} fixing`, warnings > 0 ? `${warnings} to check` : '']
      : [`${countLabel(items, 'item')} to check${isCount(summary.review_cue_count) ? ` across ${countLabel(summary.review_cue_count, 'cue')}` : ''}`]
    return {
      tone: 'error',
      headline: 'processed · attention needed',
      text: `${parts.filter(Boolean).join(' · ')}${changes}. Review before delivery.`,
    }
  }
  return null
}

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function countLabel(count: number, label: string): string {
  return `${count} ${label}${count === 1 ? '' : 's'}`
}

function formatFpsStatus(result: JobResponse['result']): string | null {
  if (!result || typeof result.fps !== 'number' || !Number.isFinite(result.fps) || result.fps <= 0) return null
  const fps = Number.isInteger(result.fps) ? result.fps.toString() : result.fps.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
  if (result.fps_source === 'explicit') return `${fps} fps selected`
  if (result.fps_source === 'fallback' || result.fps_detection_confident === false) return `${fps} fps fallback`
  if (result.fps_source === 'detected') return `${fps} fps detected`
  return `${fps} fps`
}
