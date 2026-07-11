import { Footer } from './Footer'
import { Header } from './Header'

interface Section {
  title: string
  paragraphs: string[]
}

const termsSections: Section[] = [
  { title: '1. Acceptance', paragraphs: ['By accessing or using DubSync, you agree to these Terms. If you use the service for an organization, you confirm that you can bind that organization. Do not use DubSync if you do not agree.'] },
  { title: '2. The service', paragraphs: ['DubSync synchronizes subtitle files to dialogue audio and can generate subtitle files from audio. Features may change as the service develops. The service is not an archival storage product, editing suite, or substitute for professional localization review.'] },
  { title: '3. Your content and permissions', paragraphs: ['You retain ownership of your audio, subtitles, and outputs. You grant DubSync a limited, temporary license to host, copy, transform, and send that content to subprocessors only as needed to provide, secure, and troubleshoot the requested job.', 'You must have all rights, licenses, consents, and permissions required to upload and process the content, including rights relating to copyright, performers, voices, personal data, and confidential material.'] },
  { title: '4. Automated processing and third parties', paragraphs: ['DubSync uses speech recognition, language models, hosting, and infrastructure providers. These providers process data under their own terms and privacy commitments. DubSync does not use language-model-generated timestamps; subtitle timing comes from acoustic timestamps and optional forced alignment.'] },
  { title: '5. Output review', paragraphs: ['Automated transcription, diarization, punctuation, speaker attribution, and synchronization can be incomplete or incorrect. You are responsible for reviewing outputs and QC reports before publication, broadcast, contractual delivery, or any use where an error could cause harm or loss.'] },
  { title: '6. Fees and refunds', paragraphs: ['During early access, prices are quoted before paid processing begins. Unless a written quote says otherwise, fees are based on source-audio duration and the selected workflow. Completed processing is non-refundable except where required by law or where DubSync confirms that the service failed to produce the contracted output. Subscription billing is not currently offered.'] },
  { title: '7. Acceptable use', paragraphs: ['You may not use DubSync to violate law or another person’s rights; process content without authorization; distribute malware; probe or bypass security or rate limits; overload the service; impersonate others; create unlawful surveillance; or submit content intended to exploit providers or their safety controls.'] },
  { title: '8. Retention and privacy', paragraphs: ['DubSync job files and generated artifacts are scheduled for deletion 24 hours after upload. Limited infrastructure logs or third-party provider records may follow different retention periods as described in the Privacy Policy and provider terms.'] },
  { title: '9. DubSync intellectual property', paragraphs: ['DubSync and its software, interface, documentation, branding, and service design are protected by applicable intellectual-property laws. These Terms do not transfer ownership of the service or grant permission to copy, reverse engineer, resell, or create a competing hosted service from it except where such restriction is prohibited by law.'] },
  { title: '10. Availability and termination', paragraphs: ['DubSync may suspend or terminate access to protect the service, comply with law, respond to abuse, address nonpayment, or discontinue early-access features. Processing may be interrupted by provider outages, maintenance, capacity limits, or events outside reasonable control.'] },
  { title: '11. Disclaimers', paragraphs: ['To the fullest extent permitted by law, DubSync is provided “as is” and “as available,” without warranties of uninterrupted operation, merchantability, fitness for a particular purpose, non-infringement, or error-free output. Mandatory consumer rights are not excluded.'] },
  { title: '12. Limitation of liability', paragraphs: ['To the fullest extent permitted by law, DubSync is not liable for indirect, incidental, special, consequential, exemplary, or punitive damages, or for lost revenue, profit, data, goodwill, or business opportunity. DubSync’s aggregate liability for a claim is limited to the greater of the amount you paid for the affected job or USD 100. This limit does not apply where liability cannot legally be limited.'] },
  { title: '13. Indemnity', paragraphs: ['To the extent permitted by law, you will defend and indemnify DubSync against third-party claims arising from your content, your lack of required rights or consent, your unlawful use of the service, or your material breach of these Terms.'] },
  { title: '14. Governing law and disputes', paragraphs: ['These Terms are governed by the laws of the Republic of Indonesia, without regard to conflict-of-law rules, except where mandatory local law applies. Before filing a claim, each party will try in good faith for 30 days to resolve the dispute by contacting the other party.'] },
  { title: '15. Changes and contact', paragraphs: ['DubSync may update these Terms for legal, security, or product reasons. Material changes will be posted with a new effective date. Questions may be sent to reyhanputraph@gmail.com.'] },
]

const privacySections: Section[] = [
  { title: '1. Data we process', paragraphs: ['DubSync processes the audio, subtitle files, generated outputs, QC artifacts, filenames, technical job metadata, approximate request source such as IP address, and messages you send to the support email. Do not upload personal data that is unnecessary for the job.'] },
  { title: '2. Why we process it', paragraphs: ['We use this data to provide requested subtitle processing, secure and rate-limit the service, diagnose failures, answer support requests, calculate job cost, prevent abuse, and comply with legal obligations.'] },
  { title: '3. Service providers', paragraphs: ['DubSync is designed to run primarily on Render infrastructure. Cloud processing can send content to ElevenLabs for speech recognition and Google Gemini for bounded language reasoning. Other providers are used only when configured for a specific job. Provider processing is governed by their own terms and data practices.', 'Paid Gemini API content is not used to improve Google products under Google’s published paid-service terms, but limited abuse-monitoring retention can still apply. ElevenLabs zero-retention mode is not assumed and may require an eligible enterprise arrangement.'] },
  { title: '4. Retention', paragraphs: ['Files, outputs, and job records on DubSync infrastructure are scheduled for deletion 24 hours after upload. Security and platform logs that do not contain uploaded media may be retained longer by infrastructure providers. Third-party AI providers can apply separate retention periods.'] },
  { title: '5. Legal bases and rights', paragraphs: ['Where data-protection law requires a legal basis, processing is based on performing the service you request, legitimate interests in operating and securing it, consent where requested, and legal compliance. Depending on your location, you may have rights to access, correct, delete, restrict, object, or receive a copy of personal data. Contact us to make a request.'] },
  { title: '6. International transfers', paragraphs: ['The service and its providers may process data in countries other than yours. Applicable contractual, organizational, or legal safeguards are used by the relevant provider where required.'] },
  { title: '7. Security', paragraphs: ['DubSync uses secret job links, restricted download paths, upload limits, encryption in transit, isolated job directories, security headers, and automatic deletion. No internet service can guarantee absolute security. Protect your job link because anyone who has it can access that job until it expires.'] },
  { title: '8. Cookies and analytics', paragraphs: ['The current early-access service does not use advertising trackers or behavioral analytics. Essential hosting and security systems may process request metadata. If analytics or account cookies are introduced, this policy will be updated before they are used.'] },
  { title: '9. Children', paragraphs: ['DubSync is intended for professional users and is not directed to children. Do not knowingly submit a child’s personal data without a lawful basis and all required consent.'] },
  { title: '10. Contact and changes', paragraphs: ['Privacy questions and rights requests may be sent to reyhanputraph@gmail.com. This policy may be updated as the service, providers, or legal obligations change; the effective date will be revised.'] },
]

export function LegalPage({ kind }: { kind: 'terms' | 'privacy' }) {
  const terms = kind === 'terms'
  return (
    <div className="page-shell">
      <Header />
      <main className="legal-page">
        <header><h1>{terms ? 'Terms of Service' : 'Privacy Policy'}</h1><p>Effective July 11, 2026</p></header>
        <div className="legal-content">{(terms ? termsSections : privacySections).map((section) => <section key={section.title}><h2>{section.title}</h2>{section.paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}</section>)}</div>
      </main>
      <Footer />
    </div>
  )
}
