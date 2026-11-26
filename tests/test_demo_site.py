import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_demo_site_builds(tmp_path):
    out = tmp_path / "site"
    subprocess.run(["bash", "docs/demo/build.sh", str(out)], cwd=ROOT, check=True, capture_output=True)
    html = (out / "index.html").read_text()
    # relative assets (Pages serves under /halo/), demo layer loaded before the app
    assert 'href="static/styles.css"' in html and "/static/" not in html
    assert html.index("static/demo.js") < html.index("static/app.js")
    for name in ("app.js", "styles.css", "demo.js", "demo.css", "preview.png"):
        assert (out / "static" / name).exists()
    assert (out / ".nojekyll").exists()
