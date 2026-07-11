const ACTIVE_JOB_KEY = 'dubsync:active-job'

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
