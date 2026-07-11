import { Footer } from './Footer'
import { Header } from './Header'

interface Section {
  title: string
  paragraphs: string[]
}

interface LegalDocument {
  title: string
  summary: string
  sections: Section[]
}

export type LegalKind = 'terms' | 'privacy' | 'payments'

const termsSections: Section[] = [
  { title: '1. Acceptance', paragraphs: ['By accessing or using DubSync, you agree to these Terms. If you use the service for an organization, you confirm that you can bind that organization. Do not use DubSync if you do not agree.'] },
  { title: '2. Operator and contact', paragraphs: ['DubSync is operated by Reyhan Putra as an individual business based in Indonesia. Legal, billing, privacy, and support notices may be sent to reyhanputraph@gmail.com.'] },
  { title: '3. The service', paragraphs: ['DubSync synchronizes subtitle files to dialogue audio and can generate subtitle files from audio. It is not an archival storage product, a full subtitle editor, or a substitute for professional localization review. Features may change as the service develops.'] },
  { title: '4. Your content and permissions', paragraphs: ['You retain ownership of your audio, subtitles, and outputs. You grant DubSync a limited and temporary license to host, copy, transform, and send that content to service providers only as needed to provide, secure, and troubleshoot the requested job.', 'You must hold all rights, licenses, consents, and permissions required to upload and process the content, including rights relating to copyright, performers, voices, personal data, and confidential material.'] },
  { title: '5. Automated processing and providers', paragraphs: ['DubSync uses speech recognition, language models, hosting, and infrastructure providers. These providers process data under their own terms and privacy commitments. Subtitle timing comes from acoustic timestamps and optional forced alignment. Language models do not set cue timestamps.'] },
  { title: '6. Output review', paragraphs: ['Automated transcription, diarization, punctuation, speaker attribution, and synchronization can be incomplete or incorrect. You are responsible for reviewing outputs and QC reports before publication, broadcast, contractual delivery, or any use where an error could cause harm or loss.'] },
  { title: '7. Orders, prices, and payment', paragraphs: ['DubSync does not currently offer subscriptions or automatic checkout. Paid work begins only after you accept a written quote and satisfy the payment terms stated on its invoice. Unless a quote says otherwise, prices are in USD, based on source-audio duration and workflow, and exclude applicable taxes and transfer fees.', 'The Payments and Refunds Policy forms part of these Terms and explains taxes, withholding, cancellations, reruns, and refunds.'] },
  { title: '8. Acceptable use', paragraphs: ['You may not use DubSync to violate law or another person\'s rights, process content without authorization, distribute malware, bypass security or rate limits, overload the service, impersonate others, conduct unlawful surveillance, or submit content intended to exploit a provider or its safety controls.'] },
  { title: '9. Retention and privacy', paragraphs: ['DubSync job files and generated artifacts are scheduled for deletion 24 hours after upload. Limited infrastructure logs or third-party provider records may follow different retention periods as described in the Privacy Policy and provider terms.'] },
  { title: '10. DubSync intellectual property', paragraphs: ['DubSync and its software, interface, documentation, branding, and service design are protected by applicable intellectual-property laws. These Terms do not transfer ownership of the service or grant permission to copy, reverse engineer, resell, or create a competing hosted service from it except where such restriction is prohibited by law.'] },
  { title: '11. Availability and termination', paragraphs: ['DubSync may suspend or terminate access to protect the service, comply with law, respond to abuse, address nonpayment, or discontinue a feature. Processing may be interrupted by provider outages, maintenance, capacity limits, or events outside reasonable control.'] },
  { title: '12. Disclaimers', paragraphs: ['To the fullest extent permitted by law, DubSync is provided "as is" and "as available" without warranties of uninterrupted operation, merchantability, fitness for a particular purpose, non-infringement, or error-free output. Mandatory consumer rights are not excluded.'] },
  { title: '13. Limitation of liability', paragraphs: ['To the fullest extent permitted by law, DubSync is not liable for indirect, incidental, special, consequential, exemplary, or punitive damages, or for lost revenue, profit, data, goodwill, or business opportunity. DubSync\'s aggregate liability for a claim is limited to the greater of the amount paid for the affected job or USD 100. This limit does not apply where liability cannot legally be limited.'] },
  { title: '14. Indemnity', paragraphs: ['To the extent permitted by law, you will defend and indemnify DubSync against third-party claims arising from your content, your lack of required rights or consent, your unlawful use of the service, or your material breach of these Terms.'] },
  { title: '15. Governing law and disputes', paragraphs: ['These Terms are governed by the laws of the Republic of Indonesia, without regard to conflict-of-law rules, except where mandatory local law applies. Before filing a claim, each party will try in good faith for 30 days to resolve the dispute by contacting the other party.'] },
  { title: '16. Changes and contact', paragraphs: ['DubSync may update these Terms for legal, security, or product reasons. Material changes will be posted with a new effective date. Questions may be sent to reyhanputraph@gmail.com.'] },
]

const privacySections: Section[] = [
  { title: '1. Operator and data controller', paragraphs: ['Reyhan Putra operates DubSync as an individual business based in Indonesia and acts as the controller of personal data processed for the service. Privacy questions and recorded rights requests may be sent to reyhanputraph@gmail.com.'] },
  { title: '2. Data we process', paragraphs: ['DubSync processes audio, subtitle files, generated outputs, QC artifacts, filenames, technical job metadata, approximate request source such as IP address, and messages you send to support. Do not upload personal data that is unnecessary for the job.'] },
  { title: '3. Why we process it', paragraphs: ['We use this data to provide requested subtitle processing, secure and rate-limit the service, diagnose failures, answer support requests, calculate job cost, prevent abuse, and comply with legal obligations.'] },
  { title: '4. Service providers', paragraphs: ['DubSync runs primarily on Render infrastructure. Cloud processing can send content to ElevenLabs for speech recognition and Google Gemini for bounded language reasoning. Other providers are used only when configured for a specific job. Their processing is governed by their own terms and data practices.', 'DubSync does not promise zero retention by a provider unless that setting is confirmed for the relevant account and job.'] },
  { title: '5. Retention', paragraphs: ['Files, outputs, and job records on DubSync infrastructure are scheduled for deletion 24 hours after upload. Security and platform logs that do not contain uploaded media may be retained longer by infrastructure providers. Third-party AI providers can apply separate retention periods.'] },
  { title: '6. Legal bases and rights', paragraphs: ['Where data-protection law requires a legal basis, processing is based on performing the service you request, legitimate interests in operating and securing it, consent where requested, and legal compliance. Depending on your location, you may have rights to access, correct, delete, restrict, object, or receive a copy of personal data. Send a recorded request by email to exercise a right.'] },
  { title: '7. International transfers', paragraphs: ['The service and its providers may process data in countries other than yours. The relevant provider applies its contractual, organizational, and legal safeguards where required.'] },
  { title: '8. Security', paragraphs: ['DubSync uses secret job links, restricted download paths, upload limits, encryption in transit, isolated job directories, security headers, and automatic deletion. No internet service can guarantee absolute security. Protect your job link because anyone who has it can access the job until it expires.'] },
  { title: '9. Cookies and analytics', paragraphs: ['The current service does not use advertising trackers or behavioral analytics. Essential hosting and security systems may process request metadata. If analytics or account cookies are introduced, this policy will be updated before they are used.'] },
  { title: '10. Children', paragraphs: ['DubSync is intended for professional users and is not directed to children. Do not knowingly submit a child\'s personal data without a lawful basis and all required consent.'] },
  { title: '11. Changes and contact', paragraphs: ['This policy may be updated as the service, providers, or legal obligations change. The effective date will be revised. Contact reyhanputraph@gmail.com with privacy questions or requests.'] },
]

const paymentSections: Section[] = [
  { title: '1. Manual quotes and invoices', paragraphs: ['DubSync does not currently offer subscriptions or automatic checkout. Before any paid processing, you receive a written quote that identifies the workflow, source-audio duration, price, currency, payment method, job access code, and any special delivery terms. Processing starts after cleared payment unless the quote grants written credit terms.'] },
  { title: '2. Prices and payment costs', paragraphs: ['Prices are stated in USD unless the quote uses another currency. Bank charges, payment-processor charges, and currency-conversion costs are paid by the customer unless the quote says otherwise. DubSync does not store payment-card details.'] },
  { title: '3. Taxes and withholding', paragraphs: ['Published prices and written quotes exclude applicable taxes unless they are expressly marked tax-inclusive. DubSync adds or collects VAT, sales tax, or a similar transaction tax only where legally required and shows it on the quote or invoice.', 'Tell DubSync before payment if your law requires tax withholding. The quote will state how withholding is handled. You must provide a valid official withholding certificate promptly after payment. Reyhan Putra remains responsible for the operator\'s own Indonesian tax registration, reporting, and payment obligations.'] },
  { title: '4. When a full refund applies', paragraphs: ['A full refund applies when DubSync receives your cancellation before the job enters paid provider processing, when you were charged more than once for the same job, or when DubSync cannot start the accepted job and no reasonable alternative is agreed.'] },
  { title: '5. Rerun or refund for service failure', paragraphs: ['If DubSync completes processing but fails to make the promised output available because of a confirmed DubSync or provider error, you may choose one no-cost rerun. If the rerun cannot correct the failure within five business days, you may request a refund for the affected job.'] },
  { title: '6. When a completed job is not refundable', paragraphs: ['Once paid provider processing has completed and the output is available, the job is not refundable solely because you changed your mind, supplied incorrect or poor-quality source media, lacked required rights, selected the wrong workflow or language, or expected an outcome beyond the accepted quote. This does not limit a refund required by law or the service-failure protection above.'] },
  { title: '7. How to request a refund', paragraphs: ['Email reyhanputraph@gmail.com within 7 calendar days after the affected job completes. Include the job ID, invoice reference, reason, and any useful QC evidence. Do not send API keys, payment-card details, or unnecessary personal data. DubSync will acknowledge the request within two business days and normally decide it within five business days.'] },
  { title: '8. Refund timing and method', paragraphs: ['Approved refunds are returned to the original payment method where practical. DubSync initiates the refund within 10 business days after approval. Your bank or payment provider may take additional time to post it. Currency conversion and third-party fees outside DubSync\'s control may affect the final amount received.'] },
  { title: '9. Consumer rights and contact', paragraphs: ['Nothing in this policy removes a mandatory consumer right or remedy that applies to you. Questions, cancellation notices, and complaints may be sent to Reyhan Putra at reyhanputraph@gmail.com.'] },
]

const documents: Record<LegalKind, LegalDocument> = {
  terms: {
    title: 'Terms of Service',
    summary: 'The agreement for using DubSync and submitting media for processing.',
    sections: termsSections,
  },
  privacy: {
    title: 'Privacy Policy',
    summary: 'How DubSync processes, protects, transfers, and deletes job data.',
    sections: privacySections,
  },
  payments: {
    title: 'Payments and Refunds',
    summary: 'Manual billing, tax handling, cancellations, reruns, and refund eligibility.',
    sections: paymentSections,
  },
}

export function LegalPage({ kind }: { kind: LegalKind }) {
  const document = documents[kind]
  return (
    <div className="page-shell">
      <Header />
      <main className="legal-page">
        <header className="legal-heading">
          <h1>{document.title}</h1>
          <p>{document.summary}</p>
          <span>Effective July 11, 2026</span>
        </header>
        <div className="legal-content">
          {document.sections.map((section) => (
            <section key={section.title}>
              <h2>{section.title}</h2>
              {section.paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
            </section>
          ))}
        </div>
      </main>
      <Footer />
    </div>
  )
}
