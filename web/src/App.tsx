import { useEffect, useState } from 'react'

import { loadConfig } from './api'
import { Footer } from './components/Footer'
import { Header } from './components/Header'
import { LegalPage } from './components/LegalPage'
import { MarketingSections } from './components/MarketingSections'
import { Workspace } from './components/Workspace'
import { defaultConfig, type PublicConfig } from './types'

export default function App() {
  const path = window.location.pathname.replace(/\/$/, '') || '/'
  const [config, setConfig] = useState<PublicConfig>(defaultConfig)

  useEffect(() => {
    loadConfig().then(setConfig).catch(() => setConfig(defaultConfig))
  }, [])

  if (path === '/terms') return <LegalPage kind="terms" />
  if (path === '/privacy') return <LegalPage kind="privacy" />
  if (path === '/payments') return <LegalPage kind="payments" />

  return (
    <div className="page-shell">
      <Header />
      <main>
        <Workspace config={config} />
        <MarketingSections config={config} />
      </main>
      <Footer />
    </div>
  )
}
