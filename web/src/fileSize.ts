const KIBIBYTE = 1024
const MEBIBYTE = 1024 * KIBIBYTE

export function formatBytes(bytes: number) {
  if (bytes < MEBIBYTE) return `${Math.max(1, Math.round(bytes / KIBIBYTE))} KB`
  return `${(bytes / MEBIBYTE).toFixed(1)} MB`
}

export function formatFileLimit(bytes: number) {
  if (bytes < MEBIBYTE) return `${Math.max(1, Math.floor(bytes / KIBIBYTE))} KB`
  const mebibytes = bytes / MEBIBYTE
  return `${Number.isInteger(mebibytes) ? mebibytes : mebibytes.toFixed(1)} MB`
}
