import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  clearActiveJob,
  clearActiveJobs,
  readActiveJob,
  readActiveJobs,
  writeActiveJob,
  writeActiveJobs,
} from './session'

afterEach(() => vi.restoreAllMocks())

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

  it('handles unavailable browser storage without throwing', () => {
    const onReadError = vi.fn()
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('Blocked', 'SecurityError') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('Full', 'QuotaExceededError') })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new DOMException('Blocked', 'SecurityError') })

    expect(readActiveJobs(onReadError)).toEqual([])
    expect(onReadError).toHaveBeenCalledOnce()
    expect(writeActiveJobs([{ id: 'job-1', token: 'token-1' }])).toBe(false)
    expect(clearActiveJobs()).toBe(false)
    expect(writeActiveJob({ id: 'job-1', token: 'token-1' })).toBe(false)
    expect(clearActiveJob()).toBe(false)
  })

  it('keeps legacy access recoverable when migration cannot be saved', () => {
    const legacy = { id: 'legacy-job', token: 'legacy-token' }
    sessionStorage.setItem('dubsync:active-job', JSON.stringify(legacy))
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('Full', 'QuotaExceededError') })

    expect(readActiveJobs()).toEqual([legacy])
    expect(readActiveJob()).toEqual(legacy)
  })
})
