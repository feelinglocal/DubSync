const SITE_ORIGIN = 'https://dubsync.onrender.com'
const SOCIAL_IMAGE = `${SITE_ORIGIN}/brand/dubsync-social.png`
const SOCIAL_IMAGE_ALT = 'DubSync. Timing follows the performance.'

interface RouteMetadata {
  title: string
  description: string
  canonicalPath: string
  robots: string
}

const HOME_METADATA: RouteMetadata = {
  title: 'Subtitle Sync & Audio-to-SRT for Dubbing | DubSync',
  description: 'Sync an existing SRT to dubbed dialogue audio, or generate a speaker-aware SRT from audio. Download review-ready subtitles with QC reports.',
  canonicalPath: '/',
  robots: 'index, follow, max-image-preview:large',
}

const ROUTE_METADATA: Record<string, RouteMetadata> = {
  '/': HOME_METADATA,
  '/terms': {
    title: 'Terms of Service | DubSync',
    description: 'Terms for using DubSync subtitle synchronization and audio-to-SRT processing.',
    canonicalPath: '/terms',
    robots: 'noindex, follow',
  },
  '/privacy': {
    title: 'Privacy Policy | DubSync',
    description: 'How DubSync processes, protects, transfers, and deletes subtitle job data.',
    canonicalPath: '/privacy',
    robots: 'noindex, follow',
  },
  '/payments': {
    title: 'Payments and Refunds | DubSync',
    description: 'Manual billing, tax handling, cancellations, reruns, and refund eligibility for DubSync jobs.',
    canonicalPath: '/payments',
    robots: 'noindex, follow',
  },
}

function setMeta(selector: string, attributes: Record<string, string>) {
  let element = document.head.querySelector<HTMLMetaElement>(selector)
  if (!element) {
    element = document.createElement('meta')
    document.head.append(element)
  }
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value)
}

function setCanonical(url: string) {
  let element = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')
  if (!element) {
    element = document.createElement('link')
    element.rel = 'canonical'
    document.head.append(element)
  }
  element.href = url
}

export function applyRouteMetadata(path: string) {
  const metadata = ROUTE_METADATA[path] || HOME_METADATA
  const canonical = `${SITE_ORIGIN}${metadata.canonicalPath}`
  document.title = metadata.title
  setCanonical(canonical)
  setMeta('meta[name="description"]', { name: 'description', content: metadata.description })
  setMeta('meta[name="robots"]', { name: 'robots', content: metadata.robots })
  setMeta('meta[property="og:title"]', { property: 'og:title', content: metadata.title })
  setMeta('meta[property="og:description"]', { property: 'og:description', content: metadata.description })
  setMeta('meta[property="og:url"]', { property: 'og:url', content: canonical })
  setMeta('meta[property="og:image"]', { property: 'og:image', content: SOCIAL_IMAGE })
  setMeta('meta[name="twitter:title"]', { name: 'twitter:title', content: metadata.title })
  setMeta('meta[name="twitter:description"]', { name: 'twitter:description', content: metadata.description })
  setMeta('meta[name="twitter:image"]', { name: 'twitter:image', content: SOCIAL_IMAGE })
  setMeta('meta[name="twitter:image:alt"]', { name: 'twitter:image:alt', content: SOCIAL_IMAGE_ALT })
  if (path !== '/') document.head.querySelector('script[data-home-schema]')?.remove()
}
