import { CheckCircle2, Download, FileJson, FileText, LoaderCircle, TriangleAlert } from 'lucide-react'
import { useId } from 'react'

import type { JobResponse } from '../types'

interface JobPanelProps {
  job: JobResponse
  onDownload: (kind: string) => void
  downloading: string | null
  sourceName?: string
}

export function JobPanel({ job, onDownload, downloading, sourceName }: JobPanelProps) {
  const titleId = useId()
  const complete = job.status === 'complete'
  const failed = job.status === 'failed'
  const fpsStatus = complete ? formatFpsStatus(job.result) : null
  const qcStatus = complete ? formatQcStatus(job.result) : null
  const needsReview = Boolean(qcStatus?.needsReview)
  return (
    <section className="job-panel" aria-live="polite" aria-labelledby={titleId}>
      <div className="job-summary">
        <span className={failed || needsReview ? 'status-icon is-error' : 'status-icon'} aria-hidden="true">
          {failed || needsReview ? <TriangleAlert /> : complete ? <CheckCircle2 /> : <LoaderCircle className="spin" />}
        </span>
        <div>
          <span className="section-label" id={sourceName ? undefined : titleId}>{sourceName ? 'Source' : 'Latest job'}</span>
          {sourceName && <span className="job-source-name" id={titleId}>{sourceName}</span>}
          <strong>{complete ? `${job.result?.cue_count ?? 0} cues ${needsReview ? 'processed · QC review needed' : 'ready'}` : failed ? 'Job failed' : job.status === 'processing' ? 'Processing dialogue' : 'Waiting to start'}</strong>
          <span className={failed ? 'job-message is-error' : 'job-message'}>{job.error || (complete ? 'Your result and QC files are ready.' : 'You can keep this page open while DubSync works.')}</span>
          {fpsStatus && <span className="job-detail">{fpsStatus}</span>}
          {qcStatus && <span className={needsReview ? 'job-message is-error' : 'job-detail'}>{qcStatus.text}</span>}
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
        </div>
      )}
    </section>
  )
}

function formatQcStatus(result: JobResponse['result']) {
  const summary = result?.qc_summary
  if (!summary || !isCount(summary.flags) || !isCount(summary.style_violations)) {
    return { needsReview: false, text: 'QC summary unavailable. Review the QC report.' }
  }
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
      return { needsReview: true, text: `${findings.join(' · ')}. Review the QC report before using this SRT.` }
    }
    return { needsReview: false, text: `No automated QC warnings or errors${info > 0 ? ` · ${countLabel(info, 'informational note')}` : ''}.` }
  }
  return {
    needsReview: summary.flags + summary.style_violations > 0,
    text: `${countLabel(summary.flags, 'QC flag')} · ${countLabel(summary.style_violations, 'style issue')}. Review the QC report.`,
  }
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
