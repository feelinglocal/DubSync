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
  return (
    <section className="job-panel" aria-live="polite" aria-labelledby={titleId}>
      <div className="job-summary">
        <span className={failed ? 'status-icon is-error' : 'status-icon'} aria-hidden="true">
          {complete ? <CheckCircle2 /> : failed ? <TriangleAlert /> : <LoaderCircle className="spin" />}
        </span>
        <div>
          <span className="section-label" id={sourceName ? undefined : titleId}>{sourceName ? 'Source' : 'Latest job'}</span>
          {sourceName && <span className="job-source-name" id={titleId}>{sourceName}</span>}
          <strong>{complete ? `${job.result?.cue_count ?? 0} cues ready` : failed ? 'Job failed' : job.status === 'processing' ? 'Processing dialogue' : 'Waiting to start'}</strong>
          <span>{job.error || (complete ? 'Your result and QC files are ready.' : 'You can keep this page open while DubSync works.')}</span>
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
