import { beforeEach, describe, expect, it } from 'vitest'

import {
  clearActiveJob,
  clearActiveJobs,
  readActiveJob,
  readActiveJobs,
  writeActiveJob,
  writeActiveJobs,
} from './session'

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

describe('tab-scoped batch job access', () => {
  beforeEach(() => sessionStorage.clear())

  it('stores, reads, and clears every child job access as one array', () => {
    const access = [
      { id: 'job-1', token: 'token-1' },
      { id: 'job-2', token: 'token-2' },
    ]

    writeActiveJobs(access)

    expect(readActiveJobs()).toEqual(access)
    expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual(access)
    clearActiveJobs()
    expect(readActiveJobs()).toEqual([])
    expect(sessionStorage.getItem('dubsync:active-jobs')).toBeNull()
  })

  it('returns an empty array for incomplete or corrupt batch access', () => {
    expect(readActiveJobs()).toEqual([])
    sessionStorage.setItem('dubsync:active-jobs', JSON.stringify([
      { id: 'job-1', token: 'token-1' },
      { id: 'job-only' },
    ]))
    expect(readActiveJobs()).toEqual([])
    sessionStorage.setItem('dubsync:active-jobs', '{not-json')
    expect(readActiveJobs()).toEqual([])
  })

  it('migrates legacy single-job access to the batch array format', () => {
    const legacyAccess = { id: 'legacy-job', token: 'legacy-token' }
    sessionStorage.setItem('dubsync:active-job', JSON.stringify(legacyAccess))

    expect(readActiveJobs()).toEqual([legacyAccess])
    expect(JSON.parse(sessionStorage.getItem('dubsync:active-jobs') || 'null')).toEqual([legacyAccess])
    expect(sessionStorage.getItem('dubsync:active-job')).toBeNull()
  })
})
