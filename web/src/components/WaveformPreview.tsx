import { useEffect, useRef, useState } from 'react'

interface WaveformPreviewProps {
  file: File | null
}

const MAX_WAVEFORM_BYTES = 16 * 1024 * 1024
const MAX_WAVEFORM_SECONDS = 5 * 60
const WAVEFORM_LIMIT_MESSAGE = 'Waveform preview is limited to audio up to 5 minutes and 16 MB. Audio playback and upload still work.'

export function WaveformPreview({ file }: WaveformPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaUrl = useObjectUrl(file)
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [waveformState, setWaveformState] = useState('Select audio to preview it.')
  const [metadata, setMetadata] = useState<{ file: File; duration: number } | null>(null)

  useEffect(() => {
    let cancelled = false
    setPeaks(null)
    if (!file) {
      setWaveformState('Select audio to preview it.')
      return
    }

    if (file.size > MAX_WAVEFORM_BYTES) {
      setWaveformState(WAVEFORM_LIMIT_MESSAGE)
      return
    }

    const AudioContextConstructor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextConstructor) {
      setWaveformState('Waveform preview is unavailable in this browser. Audio playback still works.')
      return
    }

    if (metadata?.file !== file) {
      setWaveformState('Loading audio information...')
      return
    }
    if (!Number.isFinite(metadata.duration) || metadata.duration <= 0) {
      setWaveformState('Waveform preview is unavailable for this audio. Audio playback still works.')
      return
    }
    if (metadata.duration > MAX_WAVEFORM_SECONDS) {
      setWaveformState(WAVEFORM_LIMIT_MESSAGE)
      return
    }

    let audioContext: AudioContext
    try {
      audioContext = new AudioContextConstructor({ sampleRate: 16_000 })
    } catch {
      setWaveformState('Waveform preview is unavailable in this browser. Audio playback still works.')
      return
    }
    setWaveformState('Reading waveform...')
    void file.arrayBuffer()
      .then((buffer) => cancelled ? null : audioContext.decodeAudioData(buffer))
      .then((decoded) => {
        if (cancelled || !decoded) return
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
  }, [file, metadata])

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
        {mediaUrl && file && <audio key={mediaUrl} controls src={mediaUrl} aria-label="Audio preview player" preload="metadata" onLoadedMetadata={(event) => setMetadata({ file, duration: event.currentTarget.duration })} onError={() => setWaveformState('Audio preview is unavailable for this file. You can still upload it for processing.')} />}
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
  const [media, setMedia] = useState<{ file: File; url: string } | null>(null)
  useEffect(() => {
    if (!file) {
      setMedia(null)
      return
    }
    const nextUrl = URL.createObjectURL(file)
    setMedia({ file, url: nextUrl })
    return () => URL.revokeObjectURL(nextUrl)
  }, [file])
  return media?.file === file ? media?.url || '' : ''
}
