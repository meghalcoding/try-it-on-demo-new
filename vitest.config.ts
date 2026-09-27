import { defineConfig } from 'vitest/config'

// Deliberately separate from vite.config.ts: that file loads vite-plugin-mkcert,
// which downloads a binary from GitHub at config time. Unit tests must not
// depend on the network or on dev certificates.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
