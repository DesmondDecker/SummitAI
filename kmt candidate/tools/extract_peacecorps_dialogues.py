"""Generic extractor for Peace Corps style manuals: 'local language | English' columns + 'word - gloss' vocabulary.
   python tools/extract_peacecorps_dialogues.py --lang temne --pdf MANUAL.pdf   (PDF with a text layer)
   python tools/extract_peacecorps_dialogues.py --lang mende --ocr-dir /tmp/ocr_out/mendemanual   (output of ocr_pages.py)
The local side must not look like English and the English side must; everything else is dropped."""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twocol import entries, load_ocr_dir, open_pdf
from clean import clean_pair, english_ratio, stop_hits, english_vocab_from

HEADERS = {"singular", "plural", "verbs", "nouns", "adjectives", "vocabulary", "lesson", "dialogue", "drill", "exercise", "english", "temne", "mende", "krio", "examples", "note"}
LABEL = re.compile(r"^\s*(?:[A-Z][A-Za-z]{0,12}(?:\s[A-Z][A-Za-z]{0,12})?\s*:|\d{1,2}[.)]|\(\w\)|[-•~*]+)\s*")
VOCAB = re.compile(r"\(?\d{1,3}\)?\s+([A-Za-z'’/\-]{2,24})\s+[-–=]\s+([A-Za-z' ,/]{2,40}?)(?=\s+\(?\d{1,3}\)?\s|\s*$)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True); ap.add_argument("--pdf"); ap.add_argument("--ocr-dir")
    ap.add_argument("--english-vocab", default="data/raw_extracts/mende_migeod.tsv"); ap.add_argument("--out", required=True); ap.add_argument("--pages", default="1-9999", help="page range to use, e.g. 8-70")
    a = ap.parse_args(); lo, hi = map(int, a.pages.split("-")); vocab = english_vocab_from(a.english_vocab)
    pages = load_ocr_dir(a.ocr_dir) if a.ocr_dir else open_pdf(a.pdf).pages
    out = []
    for pno, pg in enumerate(pages, 1):
        if not lo <= pno <= hi:
            continue
        for L, R in entries(pg):
            L, R = LABEL.sub("", L), LABEL.sub("", R)
            c = clean_pair(L, R, local_first_variant=True)
            if not c:
                continue
            l, r = c
            lw, rw = len(l.split()), len(r.split())
            if not (1 <= lw <= 10 and 1 <= rw <= 14) or stop_hits(l) > 0 or english_ratio(r, vocab) < 0.8:
                continue
            if english_ratio(l, vocab) > (0.2 if lw >= 3 else 0.5) or re.search(r"[a-z][A-Z]|\b[A-Z]{2,}\b", l):
                continue                                           # looks English, or OCR junk (random capitals inside words)
            if {w.lower() for w in re.findall(r"[A-Za-z]+", l + " " + r)} & HEADERS:
                continue
            if lw >= 4 and not (0.4 <= lw / rw <= 2.5):
                continue
            out.append((pno, l, r))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# page\t{a.lang}\tenglish   (machine-extracted from OCR; verify)\n")
        f.writelines(f"{p}\t{l}\t{r}\n" for p, l, r in dict.fromkeys(out))
    print(f"{a.lang}: {len(set(out))} pairs -> {a.out}")


if __name__ == "__main__":
    main()
