"""Build docs/index.html — the standalone watchlist page served by GitHub Pages.

The template carries a __DATA__ placeholder; this injects the scored book produced by
src/watchlist.py and wraps the result in a complete HTML document. Run after
`make watchlist`.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESET = """
<style>
  :root{color-scheme:light dark}
  body{margin:0}
  img{max-width:100%}
  [hidden]{display:none!important}
</style>"""


def build():
    tpl = (ROOT / 'docs/dashboard_template.html').read_text()
    data = (ROOT / 'outputs/dashboard_data.json').read_text()
    page = tpl.replace('__DATA__', data)
    cut = page.index('</style>') + len('</style>')
    head, rest = page[:cut], page[cut:]
    doc = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="description" content="Corporate credit watchlist - 2,723 US-listed '
        'obligors ranked by calibrated 12-month probability of default, with rating grades '
        'and SHAP reason codes.">\n'
        f'{RESET}\n{head}\n</head>\n<body>\n{rest}\n</body>\n</html>\n')
    out = ROOT / 'docs/index.html'
    out.write_text(doc)
    print(f'wrote {out} ({len(doc) / 1024:.0f} KB)')


if __name__ == '__main__':
    build()
