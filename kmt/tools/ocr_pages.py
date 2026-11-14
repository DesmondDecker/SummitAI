"""OCR scanned PDF pages to JSON word boxes (resumable; run repeatedly until it says DONE).
   python tools/ocr_pages.py scan.pdf --out /tmp/ocr_out/name --first 1 --last 92 --budget 270
Output: one JSON per page: {"width","height","words":[{"text","x0","x1","top","bottom","conf"}]}.
Word boxes (not plain text) let tools/twocol.py pair 'local | English' columns by position."""
import argparse, glob, json, os, subprocess, tempfile, time


def ocr_page(pdf, page, dpi, psm):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-f", str(page), "-l", str(page), "-png", pdf, d + "/p"], check=True, capture_output=True)
        img = glob.glob(d + "/p*.png")[0]
        subprocess.run(["tesseract", img, d + "/o", "-l", "eng", "--psm", str(psm), "tsv"], check=True, capture_output=True)
        rows = [l.rstrip("\n").split("\t") for l in open(d + "/o.tsv", encoding="utf-8")][1:]
    words, W, H = [], 0, 0
    for r in rows:
        if len(r) < 12:
            continue
        level, left, top, w, h, conf, text = int(r[0]), int(r[6]), int(r[7]), int(r[8]), int(r[9]), float(r[10]), r[11]
        if level == 1:
            W, H = w, h
        if level == 5 and text.strip() and conf >= 20:
            words.append(dict(text=text.strip(), x0=left, x1=left + w, top=top, bottom=top + h, conf=conf))
    return dict(width=W, height=H, words=words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=1); ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=170); ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--budget", type=float, default=270, help="seconds to work before stopping (resume by re-running)")
    a = ap.parse_args()
    n = int(next(l for l in subprocess.run(["pdfinfo", a.pdf], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages:")).split()[1])
    last = min(a.last or n, n)
    os.makedirs(a.out, exist_ok=True); t0 = time.time(); done = 0
    for p in range(a.first, last + 1):
        f = f"{a.out}/{p:04d}.json"
        if os.path.exists(f):
            continue
        if time.time() - t0 > a.budget:
            left = sum(1 for q in range(p, last + 1) if not os.path.exists(f"{a.out}/{q:04d}.json"))
            print(f"paused: {left} pages left; run again"); return
        json.dump(ocr_page(a.pdf, p, a.dpi, a.psm), open(f, "w"))
        done += 1
    print(f"DONE ({done} pages this run; {len(glob.glob(a.out + '/*.json'))} total)")


if __name__ == "__main__":
    main()
