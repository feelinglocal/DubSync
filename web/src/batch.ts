export const MAX_BATCH_FILES = 10

export interface SyncFilePair {
  stem: string
  audio: File
  subtitle: File
}

export interface SyncFilePairing {
  pairs: SyncFilePair[]
  error: string | null
}

export function fileStem(filename: string): string {
  const finalDot = filename.lastIndexOf('.')
  return finalDot > 0 ? filename.slice(0, finalDot) : filename
}

export function validateAudioFiles(audioFiles: readonly File[]): string | null {
  if (audioFiles.length === 0) return 'Choose between 1 and 10 audio files.'
  if (audioFiles.length > MAX_BATCH_FILES) {
    return 'Choose up to 10 audio files; batches must contain between 1 and 10 audio files.'
  }
  return duplicateStemError(audioFiles, 'audio')
}

export function pairSyncFiles(
  audioFiles: readonly File[],
  subtitleFiles: readonly File[],
): SyncFilePairing {
  const audioError = validateAudioFiles(audioFiles)
  if (audioError) return { pairs: [], error: audioError }
  if (subtitleFiles.length > MAX_BATCH_FILES) {
    return { pairs: [], error: 'Choose up to 10 subtitle files.' }
  }

  const duplicateSubtitleError = duplicateStemError(subtitleFiles, 'subtitle')
  if (duplicateSubtitleError) return { pairs: [], error: duplicateSubtitleError }

  if (subtitleFiles.length !== audioFiles.length) {
    return { pairs: [], error: 'Audio and subtitle files must use the same name before the final extension.' }
  }

  const subtitlesByStem = new Map(
    subtitleFiles.map((file) => [normalizedStem(file), file]),
  )
  const pairs = audioFiles.map((audio) => {
    const stem = fileStem(audio.name)
    const subtitle = subtitlesByStem.get(normalizedStem(audio))
    return subtitle ? { stem, audio, subtitle } : null
  })

  if (pairs.some((pair) => pair === null)) {
    return { pairs: [], error: 'Audio and subtitle files must use the same name before the final extension.' }
  }

  return { pairs: pairs as SyncFilePair[], error: null }
}

function duplicateStemError(files: readonly File[], kind: 'audio' | 'subtitle'): string | null {
  const seen = new Set<string>()
  for (const file of files) {
    const stem = normalizedStem(file)
    if (seen.has(stem)) {
      return `Duplicate ${kind} file name: ${fileStem(file.name)}. Rename files so every name is unique.`
    }
    seen.add(stem)
  }
  return null
}

function normalizedStem(file: File): string {
  return fileStem(file.name).normalize('NFKC').toUpperCase()
}
