import { chromium } from '@playwright/test'
import { mkdir, readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const webRoot = resolve(scriptDirectory, '..')
const brandDirectory = resolve(webRoot, 'public', 'brand')
const mark = await readFile(resolve(brandDirectory, 'dubsync-mark.svg'), 'utf8')
const inter = await readFile(resolve(webRoot, 'node_modules', '@fontsource-variable', 'inter', 'files', 'inter-latin-wght-normal.woff2'))
const markDataUrl = `data:image/svg+xml;base64,${Buffer.from(mark).toString('base64')}`
const interDataUrl = `data:font/woff2;base64,${Buffer.from(inter).toString('base64')}`
const MASKABLE_MARK_STYLE = 'width:64%;height:64%'

await mkdir(brandDirectory, { recursive: true })
const browser = await chromium.launch({ headless: true })

try {
  for (const [size, filename] of [
    [48, 'dubsync-icon-48.png'],
    [192, 'dubsync-icon-192.png'],
    [512, 'dubsync-icon-512.png'],
  ]) {
    const page = await browser.newPage({ viewport: { width: size, height: size } })
    await page.setContent(`<style>html,body{margin:0;width:100%;height:100%;background:transparent}img{display:block;width:100%;height:100%}</style><img src="${markDataUrl}" alt="">`)
    await page.screenshot({ path: resolve(brandDirectory, filename), omitBackground: true })
    await page.close()
  }

  for (const [size, filename, markStyle] of [
    [180, 'dubsync-apple-touch.png', 'width:72%;height:72%'],
    [192, 'dubsync-maskable-192.png', MASKABLE_MARK_STYLE],
    [512, 'dubsync-maskable-512.png', MASKABLE_MARK_STYLE],
  ]) {
    const page = await browser.newPage({ viewport: { width: size, height: size } })
    await page.setContent(`<style>html,body{margin:0;width:100%;height:100%;display:grid;place-items:center;background:#f6f8f8}img{display:block}</style><img style="${markStyle}" src="${markDataUrl}" alt="">`)
    await page.screenshot({ path: resolve(brandDirectory, filename) })
    await page.close()
  }

  const social = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 })
  await social.setContent(`
    <style>
      @font-face{font-family:"DubSync Inter";src:url("${interDataUrl}") format("woff2");font-style:normal;font-weight:100 900;font-display:block}
      *{box-sizing:border-box}
      html,body{margin:0;width:1200px;height:630px;overflow:hidden}
      body{display:grid;grid-template-columns:1fr 360px;align-items:center;gap:72px;padding:76px 88px;background:#f6f8f8;color:#091717;font-family:"DubSync Inter",ui-sans-serif,system-ui,sans-serif}
      main{align-self:center}
      .identity{display:flex;align-items:center;gap:16px;margin-bottom:54px;font-size:28px;font-weight:730;letter-spacing:-.02em}
      .identity img{width:48px;height:48px}
      h1{max-width:680px;margin:0;font-size:74px;line-height:.98;letter-spacing:-.055em;font-weight:730}
      p{max-width:620px;margin:28px 0 0;color:#586767;font-size:24px;line-height:1.4;letter-spacing:-.015em}
      .mark{width:320px;height:320px}
    </style>
    <main>
      <div class="identity"><img src="${markDataUrl}" alt=""><span>DubSync</span></div>
      <h1>Timing follows the performance.</h1>
      <p>Subtitle sync for dubbed dialogue. Audio-to-SRT with reviewable QC.</p>
    </main>
    <img class="mark" src="${markDataUrl}" alt="">
  `)
  await social.evaluate(() => document.fonts.ready.then(() => true))
  await social.screenshot({ path: resolve(brandDirectory, 'dubsync-social.png') })
  await social.close()
} finally {
  await browser.close()
}
