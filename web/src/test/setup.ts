import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

afterEach(cleanup)

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
})

Object.defineProperty(URL, 'createObjectURL', {
  configurable: true,
  value: () => 'blob:test-audio',
})

Object.defineProperty(URL, 'revokeObjectURL', {
  configurable: true,
  value: () => undefined,
})
