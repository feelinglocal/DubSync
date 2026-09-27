import { CheckCircle2, FileAudio, FileText, Trash2, Upload } from 'lucide-react'
import { useId, useRef } from 'react'

import { formatBytes, formatFileLimit } from '../fileSize'

interface BatchUploadFieldProps {
  label: string
  accept: string
  files: readonly File[]
  required?: boolean
  describedBy?: string
  kind: 'audio' | 'subtitle'
  maxBytes?: number
  onChange: (files: File[]) => void
}

export function BatchUploadField({
  label,
  accept,
  files,
  required = false,
  describedBy,
  kind,
  maxBytes = 2 * 1024 * 1024,
  onChange,
}: BatchUploadFieldProps) {
  const id = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const Icon = kind === 'audio' ? FileAudio : FileText
  const hasFiles = files.length > 0

  function clearFiles() {
    if (inputRef.current) inputRef.current.value = ''
    onChange([])
  }

  return (
    <div className="upload-field">
      <label className="upload-label" htmlFor={id}>{label}</label>
      <div className={hasFiles ? 'upload-row batch-upload-row has-file' : 'upload-row batch-upload-row'}>
        <span className="upload-icon" aria-hidden="true"><Icon /></span>
        <div className="upload-copy">
          <strong>{selectionTitle(files, kind)}</strong>
          <span>{selectionDetail(files, kind, maxBytes)}</span>
        </div>
        {hasFiles ? <CheckCircle2 className="complete-icon" aria-label="Upload selected" /> : <Upload aria-hidden="true" />}
        {hasFiles && (
          <button className="icon-button" type="button" onClick={clearFiles} aria-label={`Remove all ${label.toLowerCase()}`}>
            <Trash2 />
          </button>
        )}
        <input
          ref={inputRef}
          id={id}
          className="file-input"
          type="file"
          aria-label={label}
          accept={accept}
          required={required}
          aria-describedby={describedBy}
          multiple
          onChange={(event) => onChange(Array.from(event.target.files || []))}
        />
      </div>
      {hasFiles && (
        <ul className="batch-file-list" aria-label={`${label} files`}>
          {files.map((file, index) => <li key={`${file.name}:${file.size}:${index}`}>{file.name}</li>)}
        </ul>
      )}
    </div>
  )
}

function selectionTitle(files: readonly File[], kind: 'audio' | 'subtitle') {
  if (files.length === 0) return kind === 'audio' ? 'Choose dialogue audio' : 'Choose original SRT'
  if (files.length === 1) return files[0].name
  return `${files.length} ${kind === 'audio' ? 'audio files' : 'subtitle files'} selected`
}

function selectionDetail(files: readonly File[], kind: 'audio' | 'subtitle', maxBytes: number) {
  if (files.length === 0) {
    return kind === 'audio'
      ? 'WAV, MP3, M4A, FLAC, AAC or OGG'
      : `SRT up to ${formatFileLimit(maxBytes)} each`
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0)
  return `${formatBytes(totalBytes)} total`
}
