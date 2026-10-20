"""
Builds REPORT.pdf from REPORT.md: Markdown -> self-contained HTML (pandoc) -> A4 PDF (headless Chrome or Edge).
Uses the DejaVu fonts that ship with matplotlib, so the PDF looks the same on every machine.

Run from the repository root:  python tools/build_report_pdf.py
Needs pandoc (https://pandoc.org) and Chrome or Edge.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROWSERS = [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']


def font_faces():
    try:
        import matplotlib
        ttf = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    except ImportError:
        return ''
    faces = []
    for family, files in [('DejaVu Serif', [('DejaVuSerif', 'normal', 'normal'), ('DejaVuSerif-Bold', 'bold', 'normal'),
                                            ('DejaVuSerif-Italic', 'normal', 'italic'),
                                            ('DejaVuSerif-BoldItalic', 'bold', 'italic')]),
                          ('DejaVu Sans', [('DejaVuSans', 'normal', 'normal'), ('DejaVuSans-Bold', 'bold', 'normal')]),
                          ('DejaVu Sans Mono', [('DejaVuSansMono', 'normal', 'normal')])]:
        for name, weight, style in files:
            path = os.path.join(ttf, name + '.ttf').replace('\\', '/')
            if os.path.exists(path):
                faces.append(f"@font-face {{ font-family: '{family}'; src: url('file:///{path}'); "
                             f"font-weight: {weight}; font-style: {style}; }}")
    return '\n'.join(faces)


CSS = """
@page { size: A4; margin: 17mm 19mm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 9.6pt; line-height: 1.42; color: #1d2533;
       max-width: none; margin: 0; padding: 0; }
h1, h2, h3 { font-family: 'DejaVu Sans', Arial, sans-serif; color: #1d2533; break-after: avoid; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 10pt; }
h2 { font-size: 12pt; border-bottom: 1px solid #d5dae3; padding-bottom: 3pt; margin: 16pt 0 7pt; }
h3 { font-size: 10.5pt; margin: 12pt 0 5pt; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 20pt; }
li { margin-bottom: 2pt; }
code { font-family: 'DejaVu Sans Mono', Consolas, monospace; font-size: 8.4pt; }
a { color: #2f6fdf; }
table { border-collapse: collapse; margin: 6pt 0 9pt; font-size: 8.6pt; break-inside: avoid; }
th, td { border: 1px solid #d5dae3; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f5f9; }
figure { margin: 8pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 78%; max-height: 92mm; }
figcaption { font-size: 8.2pt; font-style: italic; color: #4a5568; margin-top: 3pt; text-align: center; }
"""


def main():
    pandoc = shutil.which('pandoc')
    browser = next((b for b in BROWSERS if os.path.exists(b)), None) or shutil.which('chrome') or shutil.which('chromium')
    if not pandoc or not browser:
        sys.exit('Needs pandoc and Chrome/Edge (found pandoc=%s, browser=%s)' % (pandoc, browser))
    tmp = tempfile.mkdtemp()
    try:
        css = os.path.join(tmp, 'report.css')
        with open(css, 'w', encoding='utf-8') as f:
            f.write(font_faces() + CSS)
        html = os.path.join(tmp, 'report.html')
        subprocess.run([pandoc, 'REPORT.md', '--standalone', '--embed-resources', '--resource-path', ROOT,
                        '--css', css, '--metadata', 'pagetitle=LucidPPN reproduction report', '-o', html],
                       cwd=ROOT, check=True)
        out = os.path.join(tmp, 'report.pdf')
        # a separate profile, so an already open Chrome window cannot take over the request
        subprocess.run([browser, '--headless=new', '--disable-gpu', '--allow-file-access-from-files',
                        f'--user-data-dir={os.path.join(tmp, "profile")}', '--no-pdf-header-footer',
                        f'--print-to-pdf={out}', 'file:///' + html.replace('\\', '/')],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        if not os.path.exists(out):
            sys.exit('The browser did not write the PDF')
        pdf = os.path.join(ROOT, os.environ.get('REPORT_PDF', 'REPORT.pdf'))
        shutil.copyfile(out, pdf)
        print('Wrote', pdf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
