"""Clean raw text into one-sentence-per-line training data.
   python tools/prepare_text.py raw_krio.txt notes/*.txt --out data/krio.txt [--append]
Normalises Unicode (NFC) and whitespace, splits sentences, drops duplicates and very short/long lines."""
import argparse, os, re, unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--append", action="store_true"); ap.add_argument("--min", type=int, default=3)
ap.add_argument("--max", type=int, default=300)
a = ap.parse_args()

seen = set()
if a.append and os.path.exists(a.out):
    seen = {l.strip() for l in open(a.out, encoding="utf-8")}
new = []
for path in a.inputs:
    text = unicodedata.normalize("NFC", open(path, encoding="utf-8", errors="ignore").read())
    for para in re.split(r"\n\s*\n|\n", text):
        for s in re.split(r"(?<=[.!?])\s+", " ".join(para.split())):
            if a.min <= len(s) <= a.max and s not in seen:
                seen.add(s); new.append(s)
with open(a.out, "a" if a.append else "w", encoding="utf-8") as f:
    f.write("".join(s + "\n" for s in new))
print(f"wrote {len(new)} sentences to {a.out}")
