import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

interface WaveformPreviewProps {
  file: File | null
}

export function WaveformPreview({ file }: WaveformPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaUrl = useObjectUrl(file)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return
    const width = canvas.width
    const height = canvas.height
    context.clearRect(0, 0, width, height)
    context.strokeStyle = '#48b89a'
    context.lineWidth = 2
    context.beginPath()
    for (let x = 0; x < width; x += 5) {
      const phase = x / width
      const amplitude = 8 + Math.abs(Math.sin(phase * 31) * 18) + Math.abs(Math.cos(phase * 13) * 10)
      context.moveTo(x, height / 2 - amplitude)
      context.lineTo(x, height / 2 + amplitude)
    }
    context.stroke()
    context.strokeStyle = '#ffc933'
    context.lineWidth = 3
    context.beginPath()
    context.moveTo(width * 0.52, 10)
    context.lineTo(width * 0.52, height - 10)
    context.stroke()
  }, [file])

  return (
    <section className={file ? 'preview-panel' : 'preview-panel is-empty'} aria-labelledby="preview-title">
      <div className="preview-heading">
        <div>
          <span className="section-label" id="preview-title">Audio preview</span>
          <strong>{file?.name || 'Your waveform will appear here'}</strong>
        </div>
        {mediaUrl && <audio controls src={mediaUrl} aria-label="Audio preview player" preload="metadata" />}
      </div>
      <canvas ref={canvasRef} width="1100" height="120" aria-label="Dialogue waveform" />
      <div className="cue-rail" aria-label="Subtitle cue preview">
        <button className="icon-button" type="button" aria-label="Previous cue"><ChevronLeft /></button>
        <div className="cue-preview"><span>00:00:08,266</span><strong>Every word has a place.</strong></div>
        <div className="cue-preview is-active"><span>00:00:10,200</span><strong>Timing follows the voice.</strong></div>
        <div className="cue-preview"><span>00:00:12,533</span><strong>Not an estimate.</strong></div>
        <button className="icon-button" type="button" aria-label="Next cue"><ChevronRight /></button>
      </div>
    </section>
  )
}

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    if (!file) {
      setUrl('')
      return
    }
    const nextUrl = URL.createObjectURL(file)
    setUrl(nextUrl)
    return () => URL.revokeObjectURL(nextUrl)
  }, [file])
  return url
}
