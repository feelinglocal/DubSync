import { AudioWaveform, Menu, X } from 'lucide-react'
import { useState } from 'react'

export function Header() {
  const [open, setOpen] = useState(false)
  function closeMenu() {
    setOpen(false)
  }
  return (
    <header className="site-header">
      <a className="brand" href="/" aria-label="DubSync home"><AudioWaveform aria-hidden="true" />DubSync</a>
      <button className="icon-button mobile-menu-button" type="button" onClick={() => setOpen((value) => !value)} aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open}>
        {open ? <X /> : <Menu />}
      </button>
      <nav className={open ? 'main-nav is-open' : 'main-nav'} aria-label="Primary navigation">
        <a href="/#workspace" onClick={closeMenu}>Workspace</a>
        <a href="/#features" onClick={closeMenu}>Features</a>
        <a href="/#pricing" onClick={closeMenu}>Pricing</a>
        <a href="/#contact" onClick={closeMenu}>Contact</a>
      </nav>
    </header>
  )
}
