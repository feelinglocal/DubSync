import { CheckCircle2, FileAudio, FileText, Trash2, Upload } from 'lucide-react'
import { useId } from 'react'

interface UploadFieldProps {
  label: string
  accept: string
  file: File | null
  required?: boolean
  kind: 'audio' | 'subtitle'
  onChange: (file: File | null) => void
}

export function UploadField({ label, accept, file, required = false, kind, onChange }: UploadFieldProps) {
  const id = useId()
  const Icon = kind === 'audio' ? FileAudio : FileText
  return (
    <div className="upload-field">
      <label className="upload-label" htmlFor={id}>{label}</label>
      <div className={file ? 'upload-row has-file' : 'upload-row'}>
        <span className="upload-icon" aria-hidden="true"><Icon /></span>
        <div className="upload-copy">
          <strong>{file?.name || (kind === 'audio' ? 'Choose dialogue audio' : 'Choose original SRT')}</strong>
          <span>{file ? formatBytes(file.size) : kind === 'audio' ? 'WAV, MP3, M4A, FLAC, AAC or OGG' : 'SRT up to 20 MB'}</span>
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

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
