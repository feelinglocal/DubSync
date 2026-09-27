(() => {
  const saved = (() => {
    try {
      return localStorage.getItem('dubsync-theme')
    } catch {
      return null
    }
  })()
  const theme = saved === 'light' || saved === 'dark'
    ? saved
    : window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#0d1418' : '#ffffff')
})()
