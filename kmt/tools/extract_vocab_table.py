"""English | Krio | Temne vocabulary table (visitsierraleone.org, S. U. Kamarah, 2006) -> one row per entry, by column position.
   python tools/extract_vocab_table.py PDF --out data/raw_extracts/krio_temne_vocab.tsv"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twocol import open_pdf
from clean import fold_marks

C1, C2 = 262, 407                     # x where the Krio and Temne columns start (measured on the PDF)


def chunks_of(row, gap=12):
    out, cur = [], [row[0]]
    for a, b in zip(row, row[1:]):
        if b["x0"] - a["x1"] >= gap:
            out.append(cur); cur = [b]
        else:
            cur.append(b)
    out.append(cur)
    return [(c[0]["x0"], " ".join(w["text"] for w in c)) for c in out]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--out", required=True); a = ap.parse_args()
    pdf = open_pdf(a.pdf); entries, started = [], False
    for pg in pdf.pages:
        rows = {}
        for w in pg.extract_words(x_tolerance=1.5, y_tolerance=2):
            if w["top"] > pg.height - 40 or "visitsierraleone" in w["text"] or w["text"].startswith("\u00a9"):
                continue
            rows.setdefault(round(w["top"] / 4), []).append(w)
        for k in sorted(rows):
            ch = chunks_of(sorted(rows[k], key=lambda w: w["x0"]))
            if not started:
                started = any(t.strip().upper() == "ENGLISH" for _, t in ch)    # table begins after its header row
                continue
            cells = ["", "", ""]
            for x, t in ch:                                      # classify each chunk by where it starts
                i = 0 if x < 200 else 1 if x < 400 else 2
                cells[i] = (cells[i] + " " + t).strip()
            if cells[0]:
                entries.append(cells)
            elif entries:                                        # wrapped continuation line
                for i in (1, 2):
                    entries[-1][i] = (entries[-1][i] + " " + cells[i]).strip()
    out = []
    for en, kr, te in entries:
        if "XYZ" in (en + kr + te).upper():
            continue
        out.append(tuple(re.sub(r"\s+", " ", x).strip() for x in (en, kr, te)))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# english\tkrio\ttemne   (Kamarah 2006, school orthography; machine-extracted by column position; verify)\n")
        f.writelines(f"{e}\t{k}\t{t}\n" for e, k, t in out if e and (k or t))
    print(f"{len(out)} entries -> {a.out}")


if __name__ == "__main__":
    main()
