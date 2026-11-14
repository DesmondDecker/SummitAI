"""Extract Mende-English pairs from F.W.H. Migeod, 'The Mende Language' (1908; public domain).
   python tools/extract_migeod.py PDF --out data/raw_extracts/mende_migeod.tsv
Works on the University-of-Toronto scan that has an OCR text layer. Output columns: part, book_page, mende, english."""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twocol import entries, open_pdf, row_chunks
from clean import clean_pair, fold_marks, tidy_punct

OFFSET = 5                                  # pdf page index = printed page + 5
PARTS = [("phrases", 17, 46), ("grammar_examples", 47, 126), ("topical_vocab", 127, 149),
         ("mende_english", 154, 178), ("english_mende", 179, 199), ("stories", 200, 284)]
_POS = re.compile(r",?\s*\b(?:tr\.?\s*(?:&|and)?\s*(?:intr\.?)?\s*v|intr\.?\s*v|v|n|adj|adv|pron|prep|conj|interj|pr|suffix|pl)\.?\b.*$", re.I)


def headword(left):
    left = left.lstrip("*†‡ ").strip()
    left = re.split(r"\s*=\s*|,", left)[0]
    return left.strip(" .;")


def gloss(right):
    right = re.split(r";|\bsee\b|\bpl\.|\bpast\b", right)[0]
    return right.strip(" .,")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pdf = open_pdf(a.pdf); rows = []; stats = {}
    for part, lo, hi in PARTS:
        raw = kept = 0
        for bp in range(lo, min(hi, len(pdf.pages) - 1 - OFFSET) + 1):
            if part == "topical_vocab":                               # 3 columns: singular | plural | English
                pairs = [(c[0], c[-1]) for c in row_chunks(pdf.pages[bp + OFFSET]) if len(c) >= 2]
            else:
                pairs = entries(pdf.pages[bp + OFFSET])
            for L, R in pairs:
                raw += 1
                if part in ("mende_english", "topical_vocab"):
                    L2, R2 = headword(L), gloss(R)
                elif part == "english_mende":
                    L2, R2 = headword(R), gloss(L)               # English headword on the left, Mende on the right
                else:
                    L2, R2 = L, R
                c = clean_pair(L2, R2)
                if c:
                    rows.append((part, bp, *c)); kept += 1
        stats[part] = (raw, kept)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# part\tbook_page\tmende\tenglish   (machine-extracted from OCR; verify before trusting)\n")
        f.writelines(f"{p}\t{bp}\t{m}\t{e}\n" for p, bp, m, e in rows)
    for part, (raw, kept) in stats.items():
        print(f"{part:<18} raw entries {raw:>5} -> kept {kept:>5}")
    print("total kept:", len(rows))


if __name__ == "__main__":
    main()
