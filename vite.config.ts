import tailwindcss from '@tailwindcss/vite';
import { nitro } from 'nitro/vite';
import vinext from 'vinext';
import { defineConfig } from 'vite';

export default defineConfig({ plugins: [tailwindcss(), vinext(), nitro()] });
