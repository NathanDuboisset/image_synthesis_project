import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
    plugins: [
        viteSingleFile(),
        {
            name: 'wgsl-loader',
            transform(src, id) {
                if (id.endsWith('.wgsl')) {
                    return { code: `export default ${JSON.stringify(src)};`, map: null };
                }
            },
        },
    ],
    build: {
        target: 'esnext',
        assetsInlineLimit: Infinity,
    },
});
