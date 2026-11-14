"""Krio (Peace Corps manual) -> (krio, english) pairs and monolingual Krio sentences, from OCR word boxes.
   python tools/ocr_pages.py krio_language_manual.pdf --out /tmp/ocr_out/krio_manual
   python tools/extract_krio_manual.py /tmp/ocr_out/krio_manual --english-vocab data/raw_extracts/mende_migeod.tsv --out data/raw_extracts"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twocol import entries, load_ocr_dir, page_rows
from clean import clean_pair, fold_marks, tidy_punct, pc_krio_to_std, english_ratio, krio_like, stop_hits

SPEAKER = re.compile(r"^\s*(?:[A-Z][A-Za-z]{0,12}(?:\s[A-Z][A-Za-z]{0,12})?\s*:|\d{1,2}[.)]|\(\w\)|[-•~]+)\s*")
LESSON_PAGES = range(9, 186)                  # pages 1-8: front matter; 186+: glossary (word - gloss lines, too noisy to pair)


def english_vocab(tsv):
    v = set()
    for l in open(tsv, encoding="utf-8"):
        if l.startswith("#"):
            continue
        v |= set(re.findall(r"[a-z']+", l.rstrip("\n").split("\t")[-1].lower()))
    return v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ocr_dir"); ap.add_argument("--english-vocab", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); vocab = english_vocab(a.english_vocab); pages = load_ocr_dir(a.ocr_dir)
    pairs, mono, raw = [], [], 0
    for pno, pg in enumerate(pages, 1):
        for L, R in (entries(pg) if pno in LESSON_PAGES else []):
            raw += 1
            if " - " in L or " - " in R or L.isupper() or R.isupper():
                continue
            L = SPEAKER.sub("", L); R = SPEAKER.sub("", R)
            L2 = pc_krio_to_std(L)
            c = clean_pair(L2, R, local_first_variant=False)
            if not c:
                continue
            k, e = c
            kw, ew = len(k.split()), len(e.split())
            if not (1 <= kw <= 14 and 1 <= ew <= 16) or k.lower() == e.lower() or not (0.45 <= kw / ew <= 2.4):
                continue
            if re.search(r"[A-Z]{3,}", k) or re.search(r"\b(KRIO|ENGLISH|TEACHER|TRAINEE)\b", k + e, re.I):
                continue
            if english_ratio(e, vocab) < 0.75 or not krio_like(k) and english_ratio(k, vocab) > 0.8:
                continue
            pairs.append((pno, k, e))
        # single-column narrative pages: monolingual Krio sentences
        rows = page_rows(pg)
        text = " ".join(" ".join(w["text"] for w in r) for r in rows)
        if pno > 8 and "ehn" in text.lower() or "dehn" in text.lower() or "foh" in text.lower():
            for s in re.split(r"(?<=[.?!])\s+", text):
                s2 = tidy_punct(pc_krio_to_std(SPEAKER.sub("", s)))
                s2 = fold_marks(s2)
                if 4 <= len(s2.split()) <= 22 and re.fullmatch(r"[A-Za-z'\-\s,.?!;:]+", s2) and krio_like(s2) and english_ratio(s2, vocab) < 0.6:
                    mono.append((pno, s2))
    pairs = list(dict.fromkeys((k.lower(), k, e) for _, k, e in pairs)); pairs = [(0, k, e) for _, k, e in pairs]
    mono = [(0, s) for s in dict.fromkeys(s for _, s in mono)]
    os.makedirs(a.out, exist_ok=True)
    with open(a.out + "/krio_manual_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("# page\tkrio\tenglish   (machine-extracted from OCR; verify)\n"); f.writelines(f"{p}\t{k}\t{e}\n" for p, k, e in pairs)
    with open(a.out + "/krio_manual_mono.tsv", "w", encoding="utf-8") as f:
        f.write("# page\tkrio sentence\n"); f.writelines(f"{p}\t{s}\n" for p, s in mono)
    print(f"raw two-column entries {raw} -> {len(pairs)} pairs kept; {len(mono)} monolingual Krio sentences")


if __name__ == "__main__":
    main()
