import { beforeEach, describe, expect, it } from 'vitest'

import { clearActiveJob, readActiveJob, writeActiveJob } from './session'

describe('tab-scoped job access', () => {
  beforeEach(() => sessionStorage.clear())

  it('returns null when access is absent, incomplete, or corrupt', () => {
    expect(readActiveJob()).toBeNull()
    sessionStorage.setItem('dubsync:active-job', JSON.stringify({ id: 'job-only' }))
    expect(readActiveJob()).toBeNull()
    sessionStorage.setItem('dubsync:active-job', '{not-json')
    expect(readActiveJob()).toBeNull()
  })

  it('writes, reads, and clears valid access', () => {
    writeActiveJob({ id: 'job-1', token: 'token-1' })
    expect(readActiveJob()).toEqual({ id: 'job-1', token: 'token-1' })
    clearActiveJob()
    expect(readActiveJob()).toBeNull()
  })
})

