import { Menu, Moon, Sun, X } from 'lucide-react'
import { useState } from 'react'

import { useTheme } from '../theme'
import { BrandLockup } from './BrandLockup'

export function Header() {
  const [open, setOpen] = useState(false)
  const { theme, toggleTheme } = useTheme()
  function closeMenu() {
    setOpen(false)
  }
  return (
    <header className="site-header">
      <BrandLockup />
      <div className="header-controls">
        <nav className={open ? 'main-nav is-open' : 'main-nav'} aria-label="Primary navigation">
          <a href="/#workspace" onClick={closeMenu}>SRT sync</a>
          <a href="/#features" onClick={closeMenu}>Subtitle QC</a>
          <a href="/#pricing" onClick={closeMenu}>Pricing</a>
          <a href="/#contact" onClick={closeMenu}>Contact</a>
        </nav>
        <button className="icon-button theme-toggle" type="button" onClick={toggleTheme} aria-label={theme === 'dark' ? 'Use light theme' : 'Use dark theme'}>
          {theme === 'dark' ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
        </button>
        <button className="icon-button mobile-menu-button" type="button" onClick={() => setOpen((value) => !value)} aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open}>
          {open ? <X aria-hidden="true" /> : <Menu aria-hidden="true" />}
        </button>
      </div>
    </header>
  )
}
