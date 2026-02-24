// This module inlines all scene data files at build time so the app works
// when opened directly as a local file (file:// protocol) without a server.
// It patches globalThis.fetch to intercept requests for known assets and
// serve them from the bundled strings instead.

import ramParams from '../data/scenes/ram/params.txt?raw';
import ramMtl from '../data/scenes/ram/ram.mtl?raw';
import ramObj from '../data/scenes/ram/ram.obj?raw';
import ramLights from '../data/scenes/ram/lights.obj?raw';
import ramVlights from '../data/scenes/ram/vlights.txt?raw';

import conferenceParams from '../data/scenes/conference/params.txt?raw';
import conferenceMtl from '../data/scenes/conference/conference.mtl?raw';
import conferenceObj from '../data/scenes/conference/conference.obj?raw';
import conferenceLights from '../data/scenes/conference/lights.obj?raw';
import conferenceVlights from '../data/scenes/conference/vlights.txt?raw';

const ASSET_MAP: Record<string, string> = {
    'data/scenes/ram/params.txt': ramParams,
    'data/scenes/ram/ram.mtl': ramMtl,
    'data/scenes/ram/ram.obj': ramObj,
    'data/scenes/ram/lights.obj': ramLights,
    'data/scenes/ram/vlights.txt': ramVlights,
    'data/scenes/conference/params.txt': conferenceParams,
    'data/scenes/conference/conference.mtl': conferenceMtl,
    'data/scenes/conference/conference.obj': conferenceObj,
    'data/scenes/conference/lights.obj': conferenceLights,
    'data/scenes/conference/vlights.txt': conferenceVlights,
};

function resolveKey(url: string): string | null {
    // Strip query string / fragment
    const clean = url.split('?')[0]!.split('#')[0]!;
    // Try suffix match against known keys
    for (const key of Object.keys(ASSET_MAP)) {
        if (clean === key || clean.endsWith('/' + key)) {
            return key;
        }
    }
    return null;
}

const originalFetch = globalThis.fetch.bind(globalThis);

globalThis.fetch = function patchedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url;
    const key = resolveKey(url);
    if (key !== null) {
        const text = ASSET_MAP[key]!;
        const response = new Response(text, {
            status: 200,
            headers: { 'Content-Type': 'text/plain; charset=utf-8' },
        });
        return Promise.resolve(response);
    }
    return originalFetch(input, init);
};
