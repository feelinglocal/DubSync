import { AudioWaveform } from 'lucide-react'

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-identity">
        <a className="brand" href="/"><AudioWaveform aria-hidden="true" />DubSync</a>
        <span>Operated by Reyhan Putra in Indonesia.</span>
      </div>
      <nav aria-label="Legal navigation">
        <a href="/terms">Terms</a>
        <a href="/privacy">Privacy</a>
        <a href="/payments">Payments</a>
        <a href="mailto:reyhanputraph@gmail.com">Contact</a>
      </nav>
    </footer>
  )
}
