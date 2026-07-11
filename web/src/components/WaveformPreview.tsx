import { useEffect, useRef, useState } from 'react'

interface WaveformPreviewProps {
  file: File | null
}

export function WaveformPreview({ file }: WaveformPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaUrl = useObjectUrl(file)
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [waveformState, setWaveformState] = useState('Select audio to preview it.')

  useEffect(() => {
    let cancelled = false
    setPeaks(null)
    if (!file) {
      setWaveformState('Select audio to preview it.')
      return
    }

    const AudioContextConstructor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextConstructor) {
      setWaveformState('Waveform preview is unavailable in this browser. Audio playback still works.')
      return
    }

    const audioContext = new AudioContextConstructor()
    setWaveformState('Reading waveform...')
    void file.arrayBuffer()
      .then((buffer) => audioContext.decodeAudioData(buffer))
      .then((decoded) => {
        if (cancelled) return
        setPeaks(downsampleWaveform(decoded.getChannelData(0), 220))
        setWaveformState(formatAudioDuration(decoded.duration))
      })
      .catch(() => {
        if (!cancelled) setWaveformState('Waveform could not be decoded. Audio playback still works.')
      })
      .finally(() => {
        void audioContext.close().catch(() => undefined)
      })

    return () => {
      cancelled = true
      void audioContext.close().catch(() => undefined)
    }
  }, [file])

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return
    const width = canvas.width
    const height = canvas.height
    context.clearRect(0, 0, width, height)
    context.strokeStyle = peaks ? '#006CFF' : '#D6DEE3'
    context.lineWidth = peaks ? 3 : 2
    context.beginPath()
    if (peaks) {
      const spacing = width / peaks.length
      peaks.forEach((peak, index) => {
        const x = (index + 0.5) * spacing
        const amplitude = Math.max(1, peak * height * 0.42)
        context.moveTo(x, height / 2 - amplitude)
        context.lineTo(x, height / 2 + amplitude)
      })
    } else {
      context.moveTo(0, height / 2)
      context.lineTo(width, height / 2)
    }
    context.stroke()
  }, [peaks])

  return (
    <section className={file ? 'preview-panel' : 'preview-panel is-empty'} aria-labelledby="preview-title">
      <div className="preview-heading">
        <div>
          <span className="section-label" id="preview-title">Audio preview</span>
          <strong>{file?.name || 'Your waveform will appear here'}</strong>
          <span className="waveform-state">{waveformState}</span>
        </div>
        {mediaUrl && <audio controls src={mediaUrl} aria-label="Audio preview player" preload="metadata" />}
      </div>
      <canvas ref={canvasRef} width="1100" height="120" aria-label="Dialogue waveform" />
    </section>
  )
}

export function downsampleWaveform(channel: Float32Array, binCount: number): number[] {
  if (binCount <= 0) return []
  if (channel.length === 0) return Array.from({ length: binCount }, () => 0)
  const bins = Math.min(binCount, channel.length)
  return Array.from({ length: bins }, (_, index) => {
    const start = Math.floor((index * channel.length) / bins)
    const end = Math.max(start + 1, Math.floor(((index + 1) * channel.length) / bins))
    let peak = 0
    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      peak = Math.max(peak, Math.abs(channel[sampleIndex] || 0))
    }
    return peak
  })
}

export function formatAudioDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Waveform ready'
  const roundedSeconds = Math.round(seconds)
  const minutes = Math.floor(roundedSeconds / 60)
  const remaining = (roundedSeconds % 60).toString().padStart(2, '0')
  return `${minutes}:${remaining} audio`
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
