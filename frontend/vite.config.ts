import tailwindcss from '@tailwindcss/vite';
import { nitro } from 'nitro/vite';
import vinext from 'vinext';
import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  // Nitro packages the production server for Vercel. In development vinext
  // already owns the RSC server; registering both causes duplicate handlers.
  plugins: [tailwindcss(), vinext(), ...(command === 'build' ? [nitro()] : [])],
}));
