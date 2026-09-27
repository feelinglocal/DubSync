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
    const value = JSON.parse(raw) as unknown
    return isActiveJobAccess(value) ? { id: value.id, token: value.token } : null
  } catch {
    return null
  }
}

export function writeActiveJob(access: ActiveJobAccess): boolean {
  return updateStorage(() => sessionStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(access)))
}

export function clearActiveJob(): boolean {
  return updateStorage(() => sessionStorage.removeItem(ACTIVE_JOB_KEY))
}

export function readActiveJobs(onStorageError?: () => void): ActiveJobAccess[] {
  let raw: string | null
  try {
    raw = sessionStorage.getItem(ACTIVE_JOBS_KEY)
  } catch {
    onStorageError?.()
    return []
  }
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
  if (writeActiveJobs([legacy])) clearActiveJob()
  return [legacy]
}

export function writeActiveJobs(accesses: readonly ActiveJobAccess[]): boolean {
  return updateStorage(() => sessionStorage.setItem(ACTIVE_JOBS_KEY, JSON.stringify(accesses)))
}

export function clearActiveJobs(): boolean {
  return updateStorage(() => sessionStorage.removeItem(ACTIVE_JOBS_KEY))
}

function updateStorage(update: () => void): boolean {
  try {
    update()
    return true
  } catch {
    return false
  }
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
