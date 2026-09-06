/**
 * Capture live GitHub Pages screenshots for the README.
 *
 *   npm install --no-save playwright
 *   CHROME_PATH=/usr/bin/google-chrome node scripts/capture-screenshots.mjs
 */
import { mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const __dirname = dirname(fileURLToPath(import.meta.url))
const ROOT = join(__dirname, '..')
const OUT_DIR = join(ROOT, 'docs', 'screenshots')
const LIVE_URL = 'https://mangeshraut712.github.io/PicoTuri-EditJudge/'

async function main() {
  await mkdir(OUT_DIR, { recursive: true })

  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  })

  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  })

  await page.goto(LIVE_URL, { waitUntil: 'networkidle', timeout: 60000 })
  await page.getByRole('heading', { name: /Algorithm Testing Suite/i }).waitFor({ timeout: 30000 })
  await page.waitForFunction(() => document.readyState === 'complete')
  await new Promise((resolve) => setTimeout(resolve, 800))

  const homePath = join(OUT_DIR, '01-home.png')
  await page.screenshot({ path: homePath, fullPage: false })
  console.log('Wrote', homePath)

  const qualityCard = page.locator('h3', { hasText: 'Quality Scorer' }).locator('xpath=ancestor::div[contains(@class,"glass")][1]')
  await qualityCard.getByRole('button', { name: /Test Now/i }).click()
  await qualityCard.getByRole('button', { name: /View Results/i }).waitFor({ timeout: 30000 })
  await qualityCard.getByRole('button', { name: /View Results/i }).click()
  await page.getByRole('heading', { name: /🎨 Quality Scorer/i }).waitFor({ timeout: 15000 })
  await page.locator('.recharts-surface').first().waitFor({ timeout: 15000 })
  await new Promise((resolve) => setTimeout(resolve, 1200))

  const featurePath = join(OUT_DIR, '02-feature.png')
  await page.screenshot({ path: featurePath, fullPage: false })
  console.log('Wrote', featurePath)

  await browser.close()
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
