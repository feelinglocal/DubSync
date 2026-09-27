import { BrandLockup } from './BrandLockup'

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-identity">
        <BrandLockup />
        <span>Part of Feels Local</span>
      </div>
      <nav aria-label="Legal navigation">
        <a href="/terms">Terms</a>
        <a href="/privacy">Privacy</a>
        <a href="/payments">Payments</a>
        <a href="mailto:rey@feelslocal.com">Contact</a>
      </nav>
    </footer>
  )
}
