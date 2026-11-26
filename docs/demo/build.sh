#!/usr/bin/env bash
# Assemble the static GitHub Pages demo from the real web app: _site/{index.html,static/*}
set -euo pipefail
out="${1:-_site}"
rm -rf "$out" && mkdir -p "$out/static"
cp src/halo/web/app.js src/halo/web/styles.css "$out/static/"
cp docs/demo/demo.js docs/demo/demo.css "$out/static/"
cp docs/assets/web-app.png "$out/static/preview.png"
# Relative asset paths (Pages serves the site under /halo/), demo layer before app.js.
sed -e 's#href="/static/styles.css">#href="static/styles.css"><link rel="stylesheet" href="static/demo.css">#' \
    -e 's#<script src="/static/app.js" defer></script>#<script src="static/demo.js"></script><script src="static/app.js" defer></script>#' \
    -e 's#<title>[^<]*</title>#<title>halo · live demo</title><meta property="og:title" content="halo: answers you can verify"><meta property="og:description" content="Corrective RAG that checks every sentence against its source. Live demo."><meta property="og:image" content="static/preview.png">#' \
    src/halo/web/index.html > "$out/index.html"
grep -q 'static/demo.js' "$out/index.html"
grep -q 'href="static/styles.css"' "$out/index.html"
touch "$out/.nojekyll"
echo "built $out"
