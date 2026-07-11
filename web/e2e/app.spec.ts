import { readFile } from 'node:fs/promises'

import { expect, test } from '@playwright/test'

test('audio-only job uploads, processes, and downloads an SRT', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Generate from audio' }).click()
  await page.getByLabel('Dialogue audio').setInputFiles({
    name: 'dialogue.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from('fixture audio'),
  })

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

test('legal pages are reachable from direct production routes', async ({ page }) => {
  await page.goto('/terms')
  await expect(page.getByRole('heading', { name: 'Terms of Service' })).toBeVisible()
  await page.goto('/privacy')
  await expect(page.getByRole('heading', { name: 'Privacy Policy' })).toBeVisible()
})

test('mobile first viewport has no horizontal overflow and introduces the next section', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(dimensions.scrollWidth).toBe(dimensions.clientWidth)
  await expect(page.getByRole('heading', { name: 'Built for subtitle professionals' })).toBeInViewport()
})
