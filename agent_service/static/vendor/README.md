# Vendored browser libraries

The chat UI renders model output as Markdown. Because that output is influenced
by retrieved corpus text — and the corpus deliberately contains prompt-injection
and jailbreak payloads — the renderer is paired with a sanitizer: parsing
untrusted text into HTML without one would turn a poisoned document into script
execution in the operator's browser.

Both files are vendored rather than loaded from a CDN. The UI is documented as
"no external dependencies, loads offline", and the deployment target is a
disconnected CTF host, so a CDN reference would break the primary use case.

| File | Library | Version | License |
| --- | --- | --- | --- |
| `marked.min.js` | [marked](https://github.com/markedjs/marked) | 15.0.7 | MIT |
| `purify.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify) | 3.2.4 | Apache-2.0 OR MPL-2.0 |

They are the unmodified upstream minified builds:

```
https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js
https://cdn.jsdelivr.net/npm/dompurify@3.2.4/dist/purify.min.js
```

`marked` — Copyright (c) 2011-2025, Christopher Jeffrey. MIT licensed.
`DOMPurify` — Copyright (c) Cure53 and other contributors. Dual licensed under
Apache License 2.0 and Mozilla Public License 2.0.

## Updating

Replace both files and the URLs above together, then re-run the UI tests. The
sanitizer is load-bearing for security: never drop it, and never widen its
allow-list to make a rendering case work — fix the renderer config instead.
