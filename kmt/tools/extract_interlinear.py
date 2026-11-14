"""Krio-English pairs from interlinear examples in a linguistics paper:
     Di man de nia faya.            <- Krio line
     DET man COP near fire          <- gloss line (ignored)
     'The man is near the fire'     <- free translation
   python tools/extract_interlinear.py paper.pdf --out data/raw_extracts/krio_vandevate.tsv"""
import argparse, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean import fold_marks, tidy_punct

TRANS = re.compile(r"^\s*[‘'’`]+\s*(.+?)\s*[’'‘`]+\s*(?:\(.*\))?\s*\d*\s*$")
LABEL = re.compile(r"^\s*(?:\(\d+\)\s*)?(?:[a-f]\.\s+)?")
GLOSS = re.compile(r"\b(DET|COP|PROG|SG|PL|PST|FUT|COMP|COMPL|NEG|DEM|POSS|OBJ|1sg|2sg|3sg|1pl|2pl|3pl)\b|\b(?:sg|pl|pst|fut|det|cop|prog)\b")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--out", required=True); a = ap.parse_args()
    text = subprocess.run(["pdftotext", "-layout", a.pdf, "-"], capture_output=True, text=True).stdout
    lines = [l.rstrip() for l in text.splitlines()]
    out = []
    for i, l in enumerate(lines):
        m = TRANS.match(l)
        if not m or i < 2:
            continue
        j = i - 1
        while j > 0 and not lines[j].strip():
            j -= 1                                           # skip blank lines
        gloss = lines[j]
        if not (GLOSS.search(gloss) or gloss.strip().islower() or len(gloss.split()) > 2):
            continue
        k = j - 1
        while k > 0 and not lines[k].strip():
            k -= 1
        krio = LABEL.sub("", lines[k]).strip()
        en = m.group(1)
        en = re.sub(r"\s*\(.*?\)\s*", " ", en); en = re.sub(r"\s+\d+$", "", en)       # drop '(Taylor-Pearce 1989:17)' and footnote digits
        krio, en = tidy_punct(fold_marks(krio)), tidy_punct(en)
        if not re.fullmatch(r"[A-Za-z'\-\s,.?!]+", krio) or "..." in krio or "..." in en or "…" in en or not (2 <= len(krio.split()) <= 20):
            continue
        if re.search(r"\b(the|is|of|was)\b", krio.lower()) and not re.search(r"\b(de|na|di|dem|den|wit|fo)\b", krio.lower()):
            continue                                         # an English sentence, not Krio
        out.append((krio, en))
    out = list(dict.fromkeys(out))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# krio\tenglish   (interlinear examples from the paper; Krio spelling normalised)\n"); f.writelines(f"{k}\t{e}\n" for k, e in out)
    print(f"{len(out)} pairs -> {a.out}")


if __name__ == "__main__":
    main()
