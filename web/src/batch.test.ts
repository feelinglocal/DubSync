import { describe, expect, it } from 'vitest'

import { pairSyncFiles } from './batch'

function audio(name: string) {
  return new File(['audio'], name, { type: 'audio/wav' })
}

function subtitle(name: string) {
  return new File(['subtitle'], name, { type: 'application/x-subrip' })
}

describe('sync batch file pairing', () => {
  it.each([1, 10])('accepts a complete batch of %i named pair(s)', (count) => {
    const audioFiles = Array.from({ length: count }, (_, index) => audio(`${index + 1}.wav`))
    const subtitleFiles = Array.from({ length: count }, (_, index) => subtitle(`${index + 1}.srt`))

    const result = pairSyncFiles(audioFiles, subtitleFiles)

    expect(result.error).toBeNull()
    expect(result.pairs).toHaveLength(count)
  })

  it('matches unordered subtitles to audio by case-insensitive stem while preserving audio order', () => {
    const firstAudio = audio('Episode-A.WAV')
    const secondAudio = audio('episode-b.wav')
    const firstSubtitle = subtitle('EPISODE-A.srt')
    const secondSubtitle = subtitle('Episode-B.SRT')

    const result = pairSyncFiles(
      [firstAudio, secondAudio],
      [secondSubtitle, firstSubtitle],
    )

    expect(result).toEqual({
      error: null,
      pairs: [
        { stem: 'Episode-A', audio: firstAudio, subtitle: firstSubtitle },
        { stem: 'episode-b', audio: secondAudio, subtitle: secondSubtitle },
      ],
    })
  })

  it('uses only the final extension when matching multi-dot names', () => {
    const audioFile = audio('show.episode.001.final.wav')
    const subtitleFile = subtitle('SHOW.EPISODE.001.FINAL.srt')

    const result = pairSyncFiles([audioFile], [subtitleFile])

    expect(result.error).toBeNull()
    expect(result.pairs).toEqual([
      { stem: 'show.episode.001.final', audio: audioFile, subtitle: subtitleFile },
    ])
  })

  it('rejects duplicate stems in either file group, including case-only duplicates', () => {
    const duplicateAudio = pairSyncFiles(
      [audio('001.wav'), audio('001.MP3')],
      [subtitle('001.srt'), subtitle('002.srt')],
    )
    const duplicateSubtitles = pairSyncFiles(
      [audio('001.wav'), audio('002.wav')],
      [subtitle('001.srt'), subtitle('001.SRT')],
    )

    expect(duplicateAudio.pairs).toEqual([])
    expect(duplicateAudio.error).toMatch(/duplicate audio file name/i)
    expect(duplicateSubtitles.pairs).toEqual([])
    expect(duplicateSubtitles.error).toMatch(/duplicate subtitle file name/i)
  })

  it('rejects empty, missing, and mismatched file sets', () => {
    const empty = pairSyncFiles([], [])
    const missingSubtitle = pairSyncFiles([audio('001.wav'), audio('002.wav')], [subtitle('001.srt')])
    const mismatchedStem = pairSyncFiles([audio('001.wav')], [subtitle('002.srt')])

    expect(empty.pairs).toEqual([])
    expect(empty.error).toMatch(/between 1 and 10 audio files/i)
    expect(missingSubtitle.pairs).toEqual([])
    expect(missingSubtitle.error).toMatch(/same name/i)
    expect(mismatchedStem.pairs).toEqual([])
    expect(mismatchedStem.error).toMatch(/same name/i)
  })

  it('rejects an eleventh audio job even when all names match', () => {
    const audioFiles = Array.from({ length: 11 }, (_, index) => audio(`${index + 1}.wav`))
    const subtitleFiles = Array.from({ length: 11 }, (_, index) => subtitle(`${index + 1}.srt`))

    const result = pairSyncFiles(audioFiles, subtitleFiles)

    expect(result.pairs).toEqual([])
    expect(result.error).toMatch(/between 1 and 10 audio files/i)
  })
})
