import { ArrowDownToLine, AudioLines, FileAudio, FileSearch, FileText, LockKeyhole, ScanLine, Upload, Users } from 'lucide-react'

import type { PublicConfig } from '../types'

export function MarketingSections({ config }: { config: PublicConfig }) {
  const pricing = config.pricing
  return (
    <>
      <section className="feature-band" id="features">
        <div className="section-inner">
          <h2>Built for subtitle professionals</h2>
          <div className="feature-list">
            <Feature icon={<ScanLine />} title="Frame-accurate timing">Cue boundaries come from acoustic word timestamps.</Feature>
            <Feature icon={<Users />} title="Speaker-aware cues">Speaker changes stay separate, even in tight exchanges.</Feature>
            <Feature icon={<FileSearch />} title="QC you can inspect">Every text change and uncertain cue stays visible.</Feature>
          </div>
        </div>
      </section>

      <section className="workflow-band">
        <div className="section-inner workflow-inner">
          <h2>From upload to delivery</h2>
          <ol className="workflow-list">
            <WorkflowStep number="1" icon={<Upload />} title="Upload">Add audio and SRT, or audio only.</WorkflowStep>
            <WorkflowStep number="2" icon={<AudioLines />} title="Process">Speech timing drives each cue.</WorkflowStep>
            <WorkflowStep number="3" icon={<ArrowDownToLine />} title="Download">Deliver SRT and QC artifacts.</WorkflowStep>
          </ol>
        </div>
      </section>

      <section className="product-pricing-band" id="pricing">
        <div className="section-inner product-pricing-inner">
          <div className="product-modes">
            <h2>Choose how you work</h2>
            <div className="mode-description"><FileText /><div><strong>Sync existing SRT</strong><span>Keep your subtitle structure and repair its timing.</span></div></div>
            <div className="mode-description"><FileAudio /><div><strong>Generate from audio</strong><span>Create a speaker-aware SRT from dialogue audio.</span></div></div>
          </div>
          <div className="pricing-table-wrap">
            <h2>Simple usage pricing</h2>
            <table>
              <thead><tr><th>Workflow</th><th>Rate</th><th>Minimum</th></tr></thead>
              <tbody>
                <PriceRow name="Audio to SRT" tier={pricing.generate} />
                <PriceRow name="Sync existing SRT" tier={pricing.sync} />
                <PriceRow name="Precision processing" tier={pricing.precision} />
              </tbody>
            </table>
            <p>Billing is coming later. Early access jobs are quoted before processing.</p>
          </div>
        </div>
      </section>

      <section className="privacy-band">
        <div className="section-inner privacy-inner">
          <div>
            <LockKeyhole />
            <h2>Your media stays yours</h2>
            <p>Files on DubSync are automatically deleted after {config.retention_hours} hours. Provider processing and retention are described in the Privacy Policy.</p>
          </div>
          <TimelineVisual />
        </div>
      </section>

      <section className="faq-band">
        <div className="section-inner faq-inner">
          <h2>Frequently asked questions</h2>
          <div>
            <Faq title="What audio formats are supported?">WAV, MP3, M4A, FLAC, AAC, and OGG dialogue tracks are accepted.</Faq>
            <Faq title="Does DubSync change my subtitle text?">Sync mode preserves unchanged cues. Spoken differences are reconciled and listed in the QC report.</Faq>
            <Faq title="Where does timing come from?">Only acoustic word timestamps and optional forced alignment. Language models never set timestamps.</Faq>
            <Faq title="How long are files retained?">Uploads and generated artifacts on DubSync are deleted after {config.retention_hours} hours.</Faq>
          </div>
        </div>
      </section>

      <section className="contact-band" id="contact">
        <div className="section-inner contact-inner">
          <div><h2>Talk to the person building it</h2><p>Questions, feedback, or a specific localization workflow?</p></div>
          <a href="mailto:reyhanputraph@gmail.com">reyhanputraph@gmail.com</a>
        </div>
      </section>
    </>
  )
}

function Feature({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <article className="feature-item"><span>{icon}</span><div><h3>{title}</h3><p>{children}</p></div></article>
}

function WorkflowStep({ number, icon, title, children }: { number: string; icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <li><span className="workflow-icon">{icon}</span><div><span className="step-number">{number}</span><h3>{title}</h3><p>{children}</p></div></li>
}

function PriceRow({ name, tier }: { name: string; tier: { usd_per_minute: number; minimum_usd: number } }) {
  return <tr><th scope="row">{name}</th><td>${tier.usd_per_minute.toFixed(2)}/min</td><td>${tier.minimum_usd.toFixed(0)} minimum</td></tr>
}

function Faq({ title, children }: { title: string; children: React.ReactNode }) {
  return <details><summary>{title}</summary><p>{children}</p></details>
}

function TimelineVisual() {
  return (
    <div className="timeline-visual" role="img" aria-label="A dialogue waveform aligned with three subtitle cues">
      <div className="timeline-top"><strong>DubSync</strong><span>00:00:12:08</span></div>
      <div className="timeline-wave" aria-hidden="true">{Array.from({ length: 80 }, (_, index) => <i key={index} style={{ height: `${8 + ((index * 17) % 34)}px` }} />)}</div>
      <div className="timeline-cues"><span>We build tools that</span><span>respect the performance,</span><span>not just the timestamps.</span></div>
    </div>
  )
}
