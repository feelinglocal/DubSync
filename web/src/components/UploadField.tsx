import { CheckCircle2, FileAudio, FileText, Trash2, Upload } from 'lucide-react'
import { useId } from 'react'

import { formatBytes, formatFileLimit } from '../fileSize'

interface UploadFieldProps {
  label: string
  accept: string
  file: File | null
  required?: boolean
  kind: 'audio' | 'subtitle' | 'style'
  maxBytes?: number
  onChange: (file: File | null) => void
}

export function UploadField({ label, accept, file, required = false, kind, maxBytes = 2 * 1024 * 1024, onChange }: UploadFieldProps) {
  const id = useId()
  const Icon = kind === 'audio' ? FileAudio : FileText
  return (
    <div className="upload-field">
      <label className="upload-label" htmlFor={id}>{label}</label>
      <div className={file ? 'upload-row has-file' : 'upload-row'}>
        <span className="upload-icon" aria-hidden="true"><Icon /></span>
        <div className="upload-copy">
          <strong>{file?.name || (kind === 'audio' ? 'Choose dialogue audio' : kind === 'style' ? 'Choose style example SRT' : 'Choose original SRT')}</strong>
          <span>{file ? formatBytes(file.size) : kind === 'audio' ? 'WAV, MP3, M4A, FLAC, AAC or OGG' : `SRT up to ${formatFileLimit(maxBytes)}`}</span>
        </div>
        {file ? <CheckCircle2 className="complete-icon" aria-label="Upload selected" /> : <Upload aria-hidden="true" />}
        {file && (
          <button className="icon-button" type="button" onClick={() => onChange(null)} aria-label={`Remove ${label.toLowerCase()}`}>
            <Trash2 />
          </button>
        )}
        <input
          id={id}
          className="file-input"
          type="file"
          aria-label={label}
          accept={accept}
          required={required}
          onChange={(event) => onChange(event.target.files?.[0] || null)}
        />
      </div>
    </div>
  )
}
