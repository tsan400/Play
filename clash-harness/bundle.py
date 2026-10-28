"""Bundle the viewer + one project's data into a single self-contained HTML.

Usage: python bundle.py [--out DIR]   (writes <slug>-clash-review.html per project)
"""
import argparse, base64, json
from pathlib import Path

HERE = Path(__file__).parent
SHIM = """<script>
window.__BUNDLE = %s;
(() => {
  const real = window.fetch;
  window.fetch = (url, opts) => {
    const key = String(url?.url ?? url).replace(/^.*data\\//, "");
    const b64 = window.__BUNDLE[key];
    if (b64 === undefined) return real(url, opts);
    const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    return Promise.resolve(new Response(bin));
  };
  addEventListener("DOMContentLoaded", () => {
    const a = document.getElementById("csv");
    new MutationObserver(() => {
      const key = a.getAttribute("href")?.replace(/^data\\//, "");
      if (key && window.__BUNDLE[key]) {
        a.href = URL.createObjectURL(new Blob([Uint8Array.from(atob(window.__BUNDLE[key]), c => c.charCodeAt(0))], { type: "text/csv" }));
        a.download = key;
      }
    }).observe(a, { attributes: true, attributeFilter: ["href"] });
  });
})();
</script>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist"))
    args = ap.parse_args()
    data = HERE / "viewer" / "data"
    html = (HERE / "viewer" / "index.html").read_text()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for proj in json.loads((data / "index.json").read_text()):
        meta = json.loads((data / f"{proj['slug']}.json").read_text())
        files = [f"{proj['slug']}.json", meta["csv"], *meta["glb"].values()]
        bundle = {f: base64.b64encode((data / f).read_bytes()).decode() for f in files}
        bundle["index.json"] = base64.b64encode(json.dumps([proj]).encode()).decode()
        page = html.replace("<script type=\"importmap\">", SHIM % json.dumps(bundle) + "<script type=\"importmap\">", 1)
        dest = out / f"{proj['slug']}-clash-review.html"
        dest.write_text(page)
        print(f"{dest}  {dest.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
