import { readFile } from 'node:fs/promises'

import { expect, test } from '@playwright/test'

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

test('two browser contexts share one access code without sharing job state', async ({ browser }) => {
  const firstContext = await browser.newContext()
  const secondContext = await browser.newContext()
  const firstPage = await firstContext.newPage()
  const secondPage = await secondContext.newPage()
  const upload = Buffer.alloc(4 * 1024 * 1024, 1)

  try {
    await Promise.all([firstPage.goto('/'), secondPage.goto('/')])
    await Promise.all([
      firstPage.getByRole('button', { name: 'Generate from audio' }).click(),
      secondPage.getByRole('button', { name: 'Generate from audio' }).click(),
    ])
    await Promise.all([
      firstPage.getByLabel('Dialogue audio').setInputFiles({
        name: 'device-a.wav',
        mimeType: 'audio/wav',
        buffer: upload,
      }),
      secondPage.getByLabel('Dialogue audio').setInputFiles({
        name: 'device-b.wav',
        mimeType: 'audio/wav',
        buffer: upload,
      }),
      firstPage.getByLabel('Job access code').fill('fixture-access-code'),
      secondPage.getByLabel('Job access code').fill('fixture-access-code'),
    ])

    await Promise.all([
      firstPage.getByRole('button', { name: 'Generate SRT' }).click(),
      secondPage.getByRole('button', { name: 'Generate SRT' }).click(),
    ])
    await Promise.all([
      expect(firstPage.getByText('2 cues ready')).toBeVisible({ timeout: 15_000 }),
      expect(secondPage.getByText('2 cues ready')).toBeVisible({ timeout: 15_000 }),
    ])

    const [firstAccess, secondAccess] = await Promise.all([
      firstPage.evaluate(() => JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || '[]') as Array<{ id: string; token: string }>),
      secondPage.evaluate(() => JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || '[]') as Array<{ id: string; token: string }>),
    ])
    expect(firstAccess).toHaveLength(1)
    expect(secondAccess).toHaveLength(1)
    expect(firstAccess[0].id).not.toBe(secondAccess[0].id)
    expect(firstAccess[0].token).not.toBe(secondAccess[0].token)

    expect((await firstContext.request.get(`/api/jobs/${firstAccess[0].id}`, {
      headers: { Authorization: `Bearer ${secondAccess[0].token}` },
    })).status()).toBe(404)
    expect((await secondContext.request.get(`/api/jobs/${secondAccess[0].id}`, {
      headers: { Authorization: `Bearer ${firstAccess[0].token}` },
    })).status()).toBe(404)
  } finally {
    await Promise.all([firstContext.close(), secondContext.close()])
  }
})

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
  await expect(page.getByText('2 cues ready')).toBeVisible({ timeout: 15_000 })

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download SRT' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('dialogue-dubsync-synced.srt')
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
  await expect(page.getByText(/cues ready/)).toBeVisible({ timeout: 15_000 })
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
    name: 'original.wav',
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
  await expect(page.getByText('2 cues ready')).toBeVisible({ timeout: 15_000 })
  const access = await page.evaluate(() => (JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || '[]') as Array<{ id: string; token: string }>)[0])
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
  expect(download.suggestedFilename()).toBe('original-dubsync-synced.srt')
  const content = await readFile(await download.path(), 'utf-8')
  expect(content).toContain('Every word has a place.')
  expect(content).toContain('Timing follows the voice.')
})

test('matched files submit as one sequential batch and keep per-source download names', async ({ page }) => {
  test.setTimeout(60_000)

  await page.goto('/')
  await expect(page.getByText('Match names: 001.wav + 001.srt. Up to 10 pairs.')).toBeVisible()

  await page.getByLabel('Dialogue audio').setInputFiles([
    { name: '001.wav', mimeType: 'audio/wav', buffer: Buffer.from('fixture audio one') },
    { name: '002.wav', mimeType: 'audio/wav', buffer: Buffer.from('fixture audio two') },
  ])
  await page.getByLabel('Original SRT').setInputFiles([
    {
      name: '002.srt',
      mimeType: 'application/x-subrip',
      buffer: Buffer.from('1\r\n00:00:10,000 --> 00:00:11,000\r\nSecond pair.\r\n'),
    },
    {
      name: '001.srt',
      mimeType: 'application/x-subrip',
      buffer: Buffer.from('1\r\n00:00:10,000 --> 00:00:11,000\r\nFirst pair.\r\n'),
    },
  ])
  await page.getByLabel('Job access code').fill('fixture-access-code')

  await page.getByRole('button', { name: 'Start sync' }).click()
  const batchResults = page.getByRole('region', { name: 'Batch results' })
  await expect(batchResults).toBeVisible()
  await expect(batchResults.getByText('001', { exact: true })).toBeVisible()
  await expect(batchResults.getByText('002', { exact: true })).toBeVisible()
  await expect(batchResults.getByText(/cues ready/)).toHaveCount(2, { timeout: 30_000 })

  const accesses = await page.evaluate(() => JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || '[]') as Array<{ id: string; token: string }>)
  expect(accesses).toHaveLength(2)
  expect(accesses.every(({ id, token }) => Boolean(id && token))).toBe(true)

  const batchDownloadPromise = page.waitForEvent('download')
  await batchResults.getByRole('button', { name: 'Download all SRTs' }).click()
  const batchDownload = await batchDownloadPromise
  expect(batchDownload.suggestedFilename()).toMatch(/^dubsync-batch-[a-f0-9]{8}-synced-srts\.zip$/)
  const archive = await readFile(await batchDownload.path())
  expect(archive.subarray(0, 4).toString('hex')).toBe('504b0304')
  expect(archive.includes(Buffer.from('001-dubsync-synced.srt'))).toBe(true)
  expect(archive.includes(Buffer.from('002-dubsync-synced.srt'))).toBe(true)

  const firstDownloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download 001 SRT' }).click()
  expect((await firstDownloadPromise).suggestedFilename()).toBe('001-dubsync-synced.srt')

  const secondDownloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download 002 SRT' }).click()
  expect((await secondDownloadPromise).suggestedFilename()).toBe('002-dubsync-synced.srt')
})

test('batch selection rejects an eleventh pair before submission', async ({ page }) => {
  await page.goto('/')
  const audioFiles = Array.from({ length: 11 }, (_, index) => ({
    name: `${String(index + 1).padStart(3, '0')}.wav`,
    mimeType: 'audio/wav',
    buffer: Buffer.from('fixture audio'),
  }))
  const subtitleFiles = Array.from({ length: 11 }, (_, index) => ({
    name: `${String(index + 1).padStart(3, '0')}.srt`,
    mimeType: 'application/x-subrip',
    buffer: Buffer.from('1\r\n00:00:00,000 --> 00:00:01,000\r\nFixture.\r\n'),
  }))

  await page.getByLabel('Dialogue audio').setInputFiles(audioFiles)
  await page.getByLabel('Original SRT').setInputFiles(subtitleFiles)

  await expect(page.getByRole('alert')).toContainText('Choose up to 10')
  await expect(page.getByRole('button', { name: 'Start sync' })).toBeDisabled()
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

test('brand, theme, and crawler surfaces use the production identity', async ({ page, request }) => {
  await page.emulateMedia({ colorScheme: 'dark' })
  await page.goto('/')

  await expect(page).toHaveTitle('Subtitle Sync & Audio-to-SRT for Dubbing | DubSync')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await expect(page.locator('header img.brand-mark')).toHaveAttribute('src', '/brand/dubsync-mark.svg')
  await expect(page.getByText('Part of Feels Local')).toBeVisible()
  await expect(
    page.getByRole('navigation', { name: 'Legal navigation' }).getByRole('link', { name: 'Contact' }),
  ).toHaveAttribute('href', 'mailto:rey@feelslocal.com')
  await page.getByRole('button', { name: 'Use light theme' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')

  const robots = await request.get('/robots.txt')
  const sitemap = await request.get('/sitemap.xml')
  const favicon = await request.get('/favicon.svg')
  const missing = await request.get('/not-a-real-page')
  expect(robots.headers()['content-type']).toContain('text/plain')
  expect((await robots.text()).startsWith('User-agent:')).toBe(true)
  expect(sitemap.headers()['content-type']).toMatch(/application\/xml|text\/xml/)
  expect(favicon.headers()['content-type']).toContain('image/svg+xml')
  expect(missing.status()).toBe(404)
})

test('mobile first viewport has no horizontal overflow and keeps the next section near the fold', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(dimensions.scrollWidth).toBe(dimensions.clientWidth)
  const pricingFits = await page.locator('.pricing-table-wrap').evaluate((element) => element.scrollWidth <= element.clientWidth)
  expect(pricingFits).toBe(true)
  const nextSectionTop = await page.locator('.feature-band').evaluate((element) => element.getBoundingClientRect().top)
  expect(nextSectionTop).toBeLessThanOrEqual(844 + 160)
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
  expect(selectGeometry).toHaveLength(3)
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

for (const viewport of [
  { width: 320, height: 900 },
  { width: 375, height: 900 },
  { width: 414, height: 900 },
  { width: 768, height: 1024 },
  { width: 1024, height: 900 },
  { width: 1440, height: 1000 },
]) {
  test(`sync controls remain contained and aligned at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    await page.goto('/')

    await expect(page.getByLabel('Maximum lines per cue')).toHaveValue('source')
    await expect(page.getByText('Use Gemini 3.5 Transcribe (testing)')).toHaveCount(0)

    const layout = await page.evaluate(() => {
      const html = document.documentElement
      const body = document.body
      const grid = document.querySelector<HTMLElement>('.workspace-options')
      const modeControls = Array.from(document.querySelectorAll<HTMLElement>('.mode-control > button')).map((element) => {
        const box = element.getBoundingClientRect()
        return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, height: box.height }
      })
      const controls = Array.from(grid?.children || []).map((element) => {
        const box = element.getBoundingClientRect()
        return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width }
      })
      return {
        clientWidth: html.clientWidth,
        scrollWidth: html.scrollWidth,
        htmlOverflowX: getComputedStyle(html).overflowX,
        bodyOverflowX: getComputedStyle(body).overflowX,
        gridWidth: grid?.getBoundingClientRect().width || 0,
        modeControls,
        controls,
      }
    })

    expect(layout.scrollWidth).toBe(layout.clientWidth)
    expect(['hidden', 'clip']).not.toContain(layout.htmlOverflowX)
    expect(['hidden', 'clip']).not.toContain(layout.bodyOverflowX)
    expect(layout.modeControls).toHaveLength(2)
    for (const control of layout.modeControls) {
      expect(control.left).toBeGreaterThanOrEqual(0)
      expect(control.right).toBeLessThanOrEqual(layout.clientWidth + 0.5)
      expect(control.height).toBeGreaterThanOrEqual(44)
    }
    const modeOverlapWidth = Math.min(layout.modeControls[0].right, layout.modeControls[1].right) - Math.max(layout.modeControls[0].left, layout.modeControls[1].left)
    const modeOverlapHeight = Math.min(layout.modeControls[0].bottom, layout.modeControls[1].bottom) - Math.max(layout.modeControls[0].top, layout.modeControls[1].top)
    expect(modeOverlapWidth > 0.5 && modeOverlapHeight > 0.5).toBe(false)
    expect(layout.controls).toHaveLength(5)
    for (const control of layout.controls) {
      expect(control.left).toBeGreaterThanOrEqual(0)
      expect(control.right).toBeLessThanOrEqual(layout.clientWidth + 0.5)
      expect(control.width).toBeGreaterThan(0)
    }
    for (let first = 0; first < layout.controls.length; first += 1) {
      for (let second = first + 1; second < layout.controls.length; second += 1) {
        const a = layout.controls[first]
        const b = layout.controls[second]
        const overlapWidth = Math.min(a.right, b.right) - Math.max(a.left, b.left)
        const overlapHeight = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top)
        expect(overlapWidth > 0.5 && overlapHeight > 0.5).toBe(false)
      }
    }

    const submitBox = await page.getByRole('button', { name: 'Start sync' }).boundingBox()
    expect(submitBox).not.toBeNull()
    if (viewport.width <= 1080) {
      expect(Math.abs((submitBox?.width || 0) - layout.gridWidth)).toBeLessThanOrEqual(1)
    } else {
      const frameRateWidth = (await page.getByLabel('Frame rate').boundingBox())?.width || 0
      const maxLinesWidth = (await page.getByLabel('Maximum lines per cue').boundingBox())?.width || 0
      expect(frameRateWidth).toBeGreaterThanOrEqual(220)
      expect(maxLinesWidth).toBeGreaterThanOrEqual(240)
    }

    if (viewport.width <= 760) {
      await page.getByRole('button', { name: 'Open menu' }).click()
      const menuBox = await page.getByRole('navigation', { name: 'Primary navigation' }).boundingBox()
      expect(menuBox).not.toBeNull()
      expect(menuBox?.x || 0).toBeGreaterThanOrEqual(0)
      expect((menuBox?.x || 0) + (menuBox?.width || 0)).toBeLessThanOrEqual(viewport.width)
    }

    await page.screenshot({
      path: testInfo.outputPath(`workspace-${viewport.width}px.png`),
      fullPage: true,
    })
  })
}

test('ten long customer filenames wrap without widening the mobile page', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 900 })
  await page.goto('/')
  await page.getByLabel('Dialogue audio').setInputFiles(Array.from({ length: 10 }, (_, index) => ({
    name: `${String(index + 1).padStart(3, '0')}-very-long-customer-episode-title-with-language-and-version.wav`,
    mimeType: 'audio/wav',
    buffer: Buffer.from('fixture audio'),
  })))

  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(dimensions.scrollWidth).toBe(dimensions.clientWidth)

  const listBox = await page.getByRole('list', { name: 'Dialogue audio files' }).boundingBox()
  expect(listBox).not.toBeNull()
  const chips = await page.getByRole('list', { name: 'Dialogue audio files' }).locator('li').evaluateAll((items) => items.map((item) => {
    const box = item.getBoundingClientRect()
    return { left: box.left, right: box.right }
  }))
  expect(chips).toHaveLength(10)
  for (const chip of chips) {
    expect(chip.left).toBeGreaterThanOrEqual((listBox?.x || 0) - 0.5)
    expect(chip.right).toBeLessThanOrEqual((listBox?.x || 0) + (listBox?.width || 0) + 0.5)
  }
})
