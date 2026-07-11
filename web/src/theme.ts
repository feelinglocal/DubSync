import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'dubsync-theme'
const DARK_QUERY = '(prefers-color-scheme: dark)'

function isTheme(value: string | null): value is Theme {
  return value === 'light' || value === 'dark'
}

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY)
    return isTheme(value) ? value : null
  } catch {
    return null
  }
}

function systemTheme() {
  return window.matchMedia?.(DARK_QUERY).matches ? 'dark' : 'light'
}

function resolvedTheme(): Theme {
  const saved = storedTheme()
  if (saved) return saved
  const bootstrapped = document.documentElement.dataset.theme ?? null
  if (isTheme(bootstrapped)) return bootstrapped
  return systemTheme()
}

function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.setAttribute(
    'content',
    theme === 'dark' ? '#0d1418' : '#ffffff',
  )
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const initialTheme = resolvedTheme()
    applyTheme(initialTheme)
    return initialTheme
  })

  useEffect(() => {
    const media = window.matchMedia?.(DARK_QUERY)
    if (!media) return
    const followSystem = (event: MediaQueryListEvent) => {
      if (storedTheme()) return
      const nextTheme = event.matches ? 'dark' : 'light'
      applyTheme(nextTheme)
      setTheme(nextTheme)
    }
    media.addEventListener('change', followSystem)
    return () => media.removeEventListener('change', followSystem)
  }, [])

  function toggleTheme() {
    const nextTheme = theme === 'dark' ? 'light' : 'dark'
    try {
      localStorage.setItem(STORAGE_KEY, nextTheme)
    } catch {
      // The in-memory preference still works when storage is unavailable.
    }
    applyTheme(nextTheme)
    setTheme(nextTheme)
  }

  return { theme, toggleTheme }
}
