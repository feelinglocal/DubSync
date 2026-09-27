import { readFile } from 'node:fs/promises'

import { expect, test } from '@playwright/test'

import { defaultConfig } from '../src/types'

function sineWaveWav(durationSeconds = 1, sampleRate = 8_000) {
  const sampleCount = durationSeconds * sampleRate
  const output = Buffer.alloc(44 + sampleCount * 2)
  output.write('RIFF', 0)
  output.writeUInt32LE(36 + sampleCount * 2, 4)
  output.write('WAVEfmt ', 8)
  output.writeUInt32LE(16, 16)
  output.writeUInt16LE(1, 20)
  output.writeUInt16LE(1, 22)
  output.writeUInt32LE(sampleRate, 24)
  output.writeUInt32LE(sampleRate * 2, 28)
  output.writeUInt16LE(2, 32)
  output.writeUInt16LE(16, 34)
  output.write('data', 36)
  output.writeUInt32LE(sampleCount * 2, 40)
  for (let index = 0; index < sampleCount; index += 1) {
    output.writeInt16LE(Math.round(Math.sin((index / sampleRate) * Math.PI * 2 * 220) * 18_000), 44 + index * 2)
  }
  return output
}

for (const width of [320, 390, 1440]) {
  for (const mode of ['sync', 'generate'] as const) {
    test(`Japanese ${mode} fixture submits and downloads at ${width}px`, async ({ page }, testInfo) => {
      const japaneseSrt = '1\n00:00:00,200 --> 00:00:01,200\n「今日は晴れです。」\n\n2\n00:00:01,400 --> 00:00:02,400\nOpenAI APIを使います。\n'
      const job = {
        id: `japanese-${mode}`, token: 'japanese-fixture-token', mode, status: 'complete', progress: 100,
        result: { cue_count: 2, cost_usd: 0 }, downloads: ['srt'], expires_at: '2099-01-01T00:00:00Z', error: null,
      }
      await page.route('**/api/config', (route) => route.fulfill({ json: { ...defaultConfig, jobs_available: true } }))
      await page.route('**/api/jobs', async (route) => {
        const request = route.request()
        expect(request.method()).toBe('POST')
        expect(request.postData()).toMatch(/name="language"\r\n\r\nja\r\n/)
        expect(request.postData()).toContain(`name="mode"\r\n\r\n${mode}\r\n`)
        await route.fulfill({ status: 202, json: job })
      })
      await page.route(`**/api/jobs/${job.id}/downloads/srt`, async (route) => {
        expect(route.request().headers().authorization).toBe(`Bearer ${job.token}`)
        await route.fulfill({
          contentType: 'application/x-subrip; charset=utf-8',
          headers: { 'content-disposition': `attachment; filename="japanese.${mode}.srt"` },
          body: japaneseSrt,
        })
      })
      await page.setViewportSize({ width, height: 900 })
      await page.goto('/')
      const language = page.getByRole('combobox', { name: 'Language' })
      await expect(language).toHaveValue('auto')
      await language.selectOption({ label: 'Japanese 日本語' })
      if (mode === 'generate') await page.getByRole('button', { name: 'Generate from audio' }).click()
      await page.getByLabel('Dialogue audio').setInputFiles({ name: '日本語.wav', mimeType: 'audio/wav', buffer: sineWaveWav() })
      if (mode === 'sync') {
        await page.getByLabel('Original SRT').setInputFiles({
          name: '日本語.srt', mimeType: 'application/x-subrip', buffer: Buffer.from(japaneseSrt),
        })
      }
      await expect(language).toHaveValue('ja')
      const selectedLabel = await language.evaluate(async (select: HTMLSelectElement) => {
        await document.fonts.ready
        const style = getComputedStyle(select)
        const context = document.createElement('canvas').getContext('2d')!
        context.font = style.font
        const label = select.selectedOptions[0].text
        const letterSpacing = Number.parseFloat(style.letterSpacing) || 0
        return {
          textWidth: context.measureText(label).width + letterSpacing * Math.max(label.length - 1, 0),
          availableWidth: select.clientWidth - Number.parseFloat(style.paddingLeft) - Number.parseFloat(style.paddingRight),
        }
      })
      expect(selectedLabel.textWidth).toBeLessThanOrEqual(selectedLabel.availableWidth)
      const submit = page.getByRole('button', { name: mode === 'sync' ? 'Start sync' : 'Generate SRT' })
      await expect(submit).toBeEnabled()
      await submit.click()
      await expect(page.getByText('2 cues ready')).toBeVisible()
      const downloadPromise = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Download SRT' }).click()
      const download = await downloadPromise
      expect(download.suggestedFilename()).toBe(`japanese.${mode}.srt`)
      expect(await readFile(await download.path(), 'utf-8')).toBe(japaneseSrt)
      const dimensions = await page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
      }))
      expect(dimensions.scrollWidth).toBe(dimensions.clientWidth)
      await page.locator('#workspace').screenshot({ path: testInfo.outputPath(`japanese-${mode}-${width}.png`) })
    })
  }
}

test('audio-only job uploads, processes, and downloads an SRT', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Generate from audio' }).click()
  await page.getByLabel('Dialogue audio').setInputFiles({
    name: 'dialogue.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from('fixture audio'),
  })
  await page.getByLabel('Job access code').fill('fixture-access-code')

  const submit = page.getByRole('button', { name: 'Generate SRT' })
  await expect(submit).toBeEnabled()
  await submit.click()
  await expect(page.getByText('2 cues ready')).toBeVisible()

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download SRT' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('dubsync.generated.srt')
  const content = await readFile(await download.path(), 'utf-8')
  expect(content).toContain('Every word has a place.')
  expect(content).toContain('Timing follows the voice.')
})

test('audio generation derives cue shape from an uploaded SRT style example', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Generate from audio' }).click()
  await page.getByRole('button', { name: 'From SRT' }).click()
  await page.getByLabel('Dialogue audio').setInputFiles({
    name: 'dialogue.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from('fixture audio'),
  })
  await page.getByLabel('Style example SRT').setInputFiles({
    name: 'compact-style.srt',
    mimeType: 'application/x-subrip',
    buffer: Buffer.from(
      '1\n00:00:00,000 --> 00:00:01,000\nCompact line\n\n' +
      '2\n00:00:01,200 --> 00:00:02,200\nShort words\n',
    ),
  })
  await page.getByLabel('Job access code').fill('fixture-access-code')

  await page.getByRole('button', { name: 'Generate SRT' }).click()
  await expect(page.getByText(/cues ready/)).toBeVisible()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download SRT' }).click()
  const download = await downloadPromise
  const content = await readFile(await download.path(), 'utf-8')
  const cueLines = content.trim().split(/\r?\n\r?\n+/).map((block) => block.split(/\r?\n/).slice(2))

  expect(cueLines.length).toBeGreaterThan(1)
  expect(cueLines.every((lines) => lines.length === 1)).toBe(true)
  expect(cueLines.flat().every((line) => line.length <= 12)).toBe(true)
})

test('sync mode survives refresh and protects job artifacts', async ({ page, request }) => {
  await page.goto('/')
  await page.getByLabel('Dialogue audio').setInputFiles({
    name: 'dialogue.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from('fixture audio'),
  })
  await page.getByLabel('Original SRT').setInputFiles({
    name: 'original.srt',
    mimeType: 'application/x-subrip',
    buffer: Buffer.from(
      '1\r\n00:00:10,000 --> 00:00:11,000\r\nEvery word has a place.\r\n\r\n' +
      '2\r\n00:00:20,000 --> 00:00:21,000\r\nTiming follows the voice.\r\n',
    ),
  })
  await page.getByLabel('Job access code').fill('fixture-access-code')

  await page.getByRole('button', { name: 'Start sync' }).click()
  await expect(page.getByText('2 cues ready')).toBeVisible()
  const access = await page.evaluate(() => JSON.parse(sessionStorage.getItem('dubsync:active-job') || '{}') as { id: string; token: string })
  expect(access.id).toBeTruthy()
  expect(access.token).toBeTruthy()

  expect((await request.get(`/api/jobs/${access.id}`)).status()).toBe(404)
  expect((await request.get(`/api/jobs/${access.id}/downloads/srt`)).status()).toBe(404)
  expect((await request.get(`/api/jobs/${access.id}`, {
    headers: { Authorization: `Bearer ${access.token}` },
  })).status()).toBe(200)

  await page.reload()
  await expect(page.getByText('2 cues ready')).toBeVisible()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download SRT' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('dubsync.synced.srt')
  const content = await readFile(await download.path(), 'utf-8')
  expect(content).toContain('Every word has a place.')
  expect(content).toContain('Timing follows the voice.')
})

test('selected audio paints a decoded nonblank waveform', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('Dialogue audio').setInputFiles({
    name: 'tone.wav',
    mimeType: 'audio/wav',
    buffer: sineWaveWav(),
  })

  await expect(page.getByText('0:01 audio')).toBeVisible()
  const bluePixels = await page.getByLabel('Dialogue waveform').evaluate((canvas: HTMLCanvasElement) => {
    const context = canvas.getContext('2d')
    if (!context) return 0
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
    let count = 0
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index] < 25 && pixels[index + 1] > 80 && pixels[index + 1] < 150 && pixels[index + 2] > 220 && pixels[index + 3] > 0) count += 1
    }
    return count
  })
  expect(bluePixels).toBeGreaterThan(100)
})

test('legal pages are reachable from direct production routes', async ({ page }) => {
  await page.goto('/terms')
  await expect(page.getByRole('heading', { name: 'Terms of Service' })).toBeVisible()
  await page.goto('/privacy')
  await expect(page.getByRole('heading', { name: 'Privacy Policy' })).toBeVisible()
  await page.goto('/payments')
  await expect(page.getByRole('heading', { name: 'Payments and Refunds' })).toBeVisible()
})

test('mobile first viewport has no horizontal overflow and introduces the next section', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(dimensions.scrollWidth).toBe(dimensions.clientWidth)
  const pricingFits = await page.locator('.pricing-table-wrap').evaluate((element) => element.scrollWidth <= element.clientWidth)
  expect(pricingFits).toBe(true)
  await expect(page.getByRole('heading', { name: 'Built for subtitle professionals' })).toBeInViewport()
})

test('workspace selects and feature rows use consistent alignment', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/')

  const selectGeometry = await page.locator('.select-control').evaluateAll((controls) => controls.map((control) => {
    const controlBox = control.getBoundingClientRect()
    const iconBox = control.querySelector('svg')?.getBoundingClientRect()
    return {
      iconInset: iconBox ? controlBox.right - iconBox.right : 0,
      iconCentered: iconBox ? Math.abs((controlBox.top + controlBox.height / 2) - (iconBox.top + iconBox.height / 2)) : 999,
    }
  }))
  expect(selectGeometry).toHaveLength(2)
  for (const geometry of selectGeometry) {
    expect(geometry.iconInset).toBeGreaterThanOrEqual(12)
    expect(geometry.iconCentered).toBeLessThanOrEqual(1)
  }

  const featureGeometry = await page.locator('.feature-item').evaluateAll((items) => items.map((item) => {
    const box = item.getBoundingClientRect()
    return { x: box.x, width: box.width }
  }))
  expect(featureGeometry).toHaveLength(4)
  expect(Math.abs(featureGeometry[0].x - featureGeometry[2].x)).toBeLessThanOrEqual(1)
  expect(Math.abs(featureGeometry[1].x - featureGeometry[3].x)).toBeLessThanOrEqual(1)
  expect(Math.max(...featureGeometry.map(({ width }) => width)) - Math.min(...featureGeometry.map(({ width }) => width))).toBeLessThanOrEqual(1)
})
