"""Helpers for reading two-column 'local language | English' pages from PDFs that have a text layer.
Pairs are built from word COORDINATES (not text order), which survives the column mix-ups of plain text extraction."""
import re, statistics
import pdfplumber


def page_rows(page, tol=4.5, top_min=30, bottom_margin=22):
    """Words of a page grouped into rows (top-to-bottom); header/footer margins dropped."""
    ws = [w for w in page.extract_words(x_tolerance=2, y_tolerance=2) if top_min < w["top"] < page.height - bottom_margin]
    ws.sort(key=lambda w: (w["top"], w["x0"]))
    rows, ref = [], None
    for w in ws:
        if ref is None or w["top"] - ref > tol:
            rows.append([]); ref = w["top"]
        rows[-1].append(w)
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def detect_split(rows, width, min_gap=14):
    """x position where the right (English) column starts, from wide whitespace gaps inside rows. None if single-column."""
    cands = []
    for r in rows:
        for a, b in zip(r, r[1:]):
            if b["x0"] - a["x1"] >= min_gap and 0.3 * width < b["x0"] < 0.75 * width:
                cands.append(b["x0"])
    if len(cands) < 3:
        return None
    cands.sort()
    # the English column's left edge is the most common start position
    buckets = {}
    for c in cands:
        buckets.setdefault(round(c / 4), []).append(c)
    best = max(buckets.values(), key=len)
    return statistics.median(best) - 5 if len(best) >= 3 else None


def join(words):
    return " ".join(w["text"] for w in words)


def entries(page, indent=7, min_gap=10):
    """Yield (left_text, right_text) entries of a two-column page.
    A left line starting at the left margin begins a new entry; indented left lines and any right-only lines
    continue the current one. Rows that run across the column boundary are prose and are skipped."""
    rows = page_rows(page)
    split = detect_split(rows, page.width)
    if split is None:
        return []
    lefts = [min(w["x0"] for w in r if w["x0"] < split) for r in rows if any(w["x0"] < split for w in r)]
    if not lefts:
        return []
    margin = sorted(lefts)[max(0, len(lefts) // 10)]
    out, cur = [], None
    for r in rows:
        L = [w for w in r if w["x0"] < split]; R = [w for w in r if w["x0"] >= split]
        if L and R and R[0]["x0"] - L[-1]["x1"] < min_gap:          # text crosses the boundary: a prose line
            cur = None; continue
        if L and L[-1]["x1"] > split + 5:
            cur = None; continue
        if L:
            if L[0]["x0"] <= margin + indent or cur is None:
                cur = [join(L), join(R) if R else ""]; out.append(cur)
            else:
                cur[0] += " " + join(L)
                if R: cur[1] += " " + join(R)
        elif R and cur is not None:
            cur[1] += " " + join(R)
    return [(a, b) for a, b in out]


def row_chunks(page, gap=12):
    """Each row split into chunks wherever there is >= `gap` points of whitespace: [[text, ...], ...] per row.
    For 3-column vocabulary pages: first chunk = headword, last chunk = English gloss."""
    out = []
    for r in page_rows(page):
        chunks, cur = [], [r[0]]
        for a, b in zip(r, r[1:]):
            if b["x0"] - a["x1"] >= gap:
                chunks.append(cur); cur = [b]
            else:
                cur.append(b)
        chunks.append(cur)
        out.append([join(c) for c in chunks])
    return out


class JsonPage:
    """Adapter so OCR output (tools/ocr_pages.py) can be used wherever a pdfplumber page is expected.
    Pixel coordinates are converted to PDF points so the same distance thresholds apply."""
    def __init__(self, path, dpi=170):
        import json
        d = json.load(open(path)); k = 72.0 / dpi
        self.width, self.height = d["width"] * k, d["height"] * k
        self._w = [dict(text=w["text"], x0=w["x0"] * k, x1=w["x1"] * k, top=w["top"] * k, bottom=w["bottom"] * k) for w in d["words"]]

    def extract_words(self, **kw):
        return [dict(w) for w in self._w]


def load_ocr_dir(path, dpi=170):
    import glob
    return [JsonPage(f, dpi) for f in sorted(glob.glob(path + "/*.json"))]


def open_pdf(path):
    return pdfplumber.open(path)
