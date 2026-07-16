const ACTIVE_JOB_KEY = 'dubsync:active-job'
const ACTIVE_JOBS_KEY = 'dubsync:active-jobs'

export interface ActiveJobAccess {
  id: string
  token: string
}

export function readActiveJob(): ActiveJobAccess | null {
  try {
    const raw = sessionStorage.getItem(ACTIVE_JOB_KEY)
    if (!raw) return null
    const value = JSON.parse(raw) as Partial<ActiveJobAccess>
    return value.id && value.token ? { id: value.id, token: value.token } : null
  } catch {
    return null
  }
}

export function writeActiveJob(access: ActiveJobAccess): void {
  sessionStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(access))
}

export function clearActiveJob(): void {
  sessionStorage.removeItem(ACTIVE_JOB_KEY)
}

export function readActiveJobs(): ActiveJobAccess[] {
  const raw = sessionStorage.getItem(ACTIVE_JOBS_KEY)
  if (raw) {
    try {
      const value = JSON.parse(raw) as unknown
      return isActiveJobAccessArray(value) ? value : []
    } catch {
      return []
    }
  }

  const legacy = readActiveJob()
  if (!legacy) return []
  writeActiveJobs([legacy])
  clearActiveJob()
  return [legacy]
}

export function writeActiveJobs(accesses: readonly ActiveJobAccess[]): void {
  sessionStorage.setItem(ACTIVE_JOBS_KEY, JSON.stringify(accesses))
}

export function clearActiveJobs(): void {
  sessionStorage.removeItem(ACTIVE_JOBS_KEY)
}

function isActiveJobAccessArray(value: unknown): value is ActiveJobAccess[] {
  return Array.isArray(value)
    && value.length > 0
    && value.every((access) => isActiveJobAccess(access))
}

function isActiveJobAccess(value: unknown): value is ActiveJobAccess {
  if (!value || typeof value !== 'object') return false
  const access = value as Partial<ActiveJobAccess>
  return typeof access.id === 'string' && access.id.length > 0
    && typeof access.token === 'string' && access.token.length > 0
}
