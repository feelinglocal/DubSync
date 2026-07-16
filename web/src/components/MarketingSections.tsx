import { ArrowDownToLine, AudioLines, FileAudio, FileSearch, FileText, LockKeyhole, ScanLine, Upload, Users } from 'lucide-react'

import type { PublicConfig } from '../types'

export function MarketingSections({ config }: { config: PublicConfig }) {
  const pricing = config.pricing
  return (
    <>
      <section className="feature-band" id="features">
        <div className="section-inner">
          <header className="section-heading">
            <h2>Professional subtitle sync and auto captioning</h2>
            <p>Keep valid source structure, follow the recorded performance, and review every uncertain change.</p>
          </header>
          <div className="feature-list">
            <Feature icon={<ScanLine />} title="Frame-accurate timing">Cue boundaries come from acoustic word timestamps and optional forced alignment.</Feature>
            <Feature icon={<Users />} title="Speaker-aware cues">Speaker changes remain separate in close exchanges and overlapping dialogue.</Feature>
            <Feature icon={<FileSearch />} title="Reviewable changes">Every text decision, uncertain cue, and timing warning appears in QC artifacts.</Feature>
            <Feature icon={<FileAudio />} title="Works with or without an SRT">Sync a supplied subtitle file, or create a new speaker-aware SRT directly from audio.</Feature>
          </div>
        </div>
      </section>

      <section className="workflow-band">
        <div className="section-inner workflow-inner">
          <header className="section-heading">
            <h2>From source to delivery</h2>
            <p>A direct workflow with protected job access and downloadable review artifacts.</p>
          </header>
          <ol className="workflow-list">
            <WorkflowStep icon={<Upload />} title="Upload your source">Add dialogue audio with an SRT, or choose audio-only generation.</WorkflowStep>
            <WorkflowStep icon={<AudioLines />} title="Process the performance">Speech timing drives cue boundaries while language passes handle text decisions.</WorkflowStep>
            <WorkflowStep icon={<ArrowDownToLine />} title="Download and review">Get the final SRT plus machine-readable and browser-ready QC reports.</WorkflowStep>
          </ol>
        </div>
      </section>

      <section className="product-pricing-band" id="pricing">
        <div className="section-inner product-pricing-inner">
          <div className="product-modes">
            <h2>Two focused workflows</h2>
            <div className="mode-description"><FileText /><div><strong>Sync existing SRT</strong><span>Keep the subtitle structure and repair its timing against the performance.</span></div></div>
            <div className="mode-description"><FileAudio /><div><strong>Generate from audio</strong><span>Create a speaker-aware SRT when no original subtitle file exists.</span></div></div>
          </div>
          <div className="pricing-table-wrap">
            <h2>Usage pricing</h2>
            <table>
              <thead><tr><th>Workflow</th><th>Rate</th><th>Minimum</th></tr></thead>
              <tbody>
                <PriceRow name="Audio to SRT" tier={pricing.generate} />
                <PriceRow name="Sync existing SRT" tier={pricing.sync} />
                <PriceRow name="Precision processing" tier={pricing.precision} />
              </tbody>
            </table>
            <p>Manual quote and invoice before paid processing. Accepted quotes receive a job access code. Prices exclude applicable taxes unless the quote states otherwise.</p>
            <a className="text-link" href="/payments">Read the payment and refund policy</a>
          </div>
        </div>
      </section>

      <section className="privacy-band">
        <div className="section-inner privacy-inner">
          <header className="privacy-heading">
            <LockKeyhole />
            <div><h2>Your media stays yours</h2><p>Short retention, protected job access, and no advertising trackers.</p></div>
          </header>
          <dl className="privacy-facts">
            <div><dt>{config.retention_hours} hours</dt><dd>Uploads and generated artifacts are scheduled for deletion.</dd></div>
            <div><dt>Secret access</dt><dd>Each job uses a browser-held token for status and downloads.</dd></div>
            <div><dt>Named providers</dt><dd>Render, ElevenLabs, and Gemini processing is disclosed in the Privacy Policy.</dd></div>
          </dl>
        </div>
      </section>

      <section className="faq-band">
        <div className="section-inner faq-inner">
          <h2>Frequently asked questions</h2>
          <div>
            <Faq title="What audio formats are supported?">WAV, MP3, M4A, FLAC, AAC, and OGG dialogue tracks are accepted.</Faq>
            <Faq title="Does DubSync change my subtitle text?">Sync mode preserves unchanged cues. Spoken differences are reconciled and listed in the QC report.</Faq>
            <Faq title="Where does timing come from?">Only acoustic word timestamps and optional forced alignment. Language models never set timestamps.</Faq>
            <Faq title="How long are files retained?">Uploads and generated artifacts on DubSync are scheduled for deletion after {config.retention_hours} hours.</Faq>
            <Faq title="Can DubSync create automatic captions?">Audio-to-SRT mode creates dialogue subtitles from audio. Add non-speech sound descriptions during review when accessibility captions require them.</Faq>
          </div>
        </div>
      </section>

      <section className="contact-band" id="contact">
        <div className="section-inner contact-inner">
          <div><h2>Contact the operator</h2><p>Questions, billing requests, feedback, or a specific localization workflow.</p></div>
          <a href="mailto:rey@feelslocal.com">rey@feelslocal.com</a>
        </div>
      </section>
    </>
  )
}

function Feature({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <article className="feature-item"><span>{icon}</span><div><h3>{title}</h3><p>{children}</p></div></article>
}

function WorkflowStep({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <li><span className="workflow-icon">{icon}</span><div><h3>{title}</h3><p>{children}</p></div></li>
}

function PriceRow({ name, tier }: { name: string; tier: { usd_per_minute: number; minimum_usd: number } }) {
  return <tr><th scope="row">{name}</th><td>${tier.usd_per_minute.toFixed(2)}/min</td><td>${tier.minimum_usd.toFixed(0)} minimum</td></tr>
}

function Faq({ title, children }: { title: string; children: React.ReactNode }) {
  return <details><summary>{title}</summary><p>{children}</p></details>
}
