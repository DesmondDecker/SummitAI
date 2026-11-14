"""Turn a Bible in language X plus an English Bible into KMT training data.

  python tools/bible_align.py --lang krio --src krio_bible.usfm --english web_english/ --data data

Inputs (files or folders) may be USFM/SFM, or plain text with lines like
  "Genesis 1:1 In di bigin..."   "GEN 1:1<TAB>text"   or   "1:1 text" (one file per book).
If the book names in a file are not English (e.g. Krio names), books are matched BY ORDER, so the file
must be in the normal Bible order. The mapping is printed so you can check it.

Writes:  data/parallel/<lang>-en.tsv   aligned verse pairs (training)
         data/heldout/<lang>-en.tsv    whole books kept out for honest testing (--holdout)
         data/<lang>.txt               verses with no usable English match (extra monolingual text)
"""
import argparse, glob, os, re, shutil, sys, unicodedata

CANON = [("GEN", "Genesis"), ("EXO", "Exodus"), ("LEV", "Leviticus"), ("NUM", "Numbers"), ("DEU", "Deuteronomy"),
         ("JOS", "Joshua"), ("JDG", "Judges"), ("RUT", "Ruth"), ("1SA", "1 Samuel"), ("2SA", "2 Samuel"),
         ("1KI", "1 Kings"), ("2KI", "2 Kings"), ("1CH", "1 Chronicles"), ("2CH", "2 Chronicles"), ("EZR", "Ezra"),
         ("NEH", "Nehemiah"), ("EST", "Esther"), ("JOB", "Job"), ("PSA", "Psalms"), ("PRO", "Proverbs"),
         ("ECC", "Ecclesiastes"), ("SNG", "Song of Solomon"), ("ISA", "Isaiah"), ("JER", "Jeremiah"),
         ("LAM", "Lamentations"), ("EZK", "Ezekiel"), ("DAN", "Daniel"), ("HOS", "Hosea"), ("JOL", "Joel"),
         ("AMO", "Amos"), ("OBA", "Obadiah"), ("JON", "Jonah"), ("MIC", "Micah"), ("NAM", "Nahum"),
         ("HAB", "Habakkuk"), ("ZEP", "Zephaniah"), ("HAG", "Haggai"), ("ZEC", "Zechariah"), ("MAL", "Malachi"),
         ("MAT", "Matthew"), ("MRK", "Mark"), ("LUK", "Luke"), ("JHN", "John"), ("ACT", "Acts"), ("ROM", "Romans"),
         ("1CO", "1 Corinthians"), ("2CO", "2 Corinthians"), ("GAL", "Galatians"), ("EPH", "Ephesians"),
         ("PHP", "Philippians"), ("COL", "Colossians"), ("1TH", "1 Thessalonians"), ("2TH", "2 Thessalonians"),
         ("1TI", "1 Timothy"), ("2TI", "2 Timothy"), ("TIT", "Titus"), ("PHM", "Philemon"), ("HEB", "Hebrews"),
         ("JAS", "James"), ("1PE", "1 Peter"), ("2PE", "2 Peter"), ("1JN", "1 John"), ("2JN", "2 John"),
         ("3JN", "3 John"), ("JUD", "Jude"), ("REV", "Revelation")]
INDEX = {c: i for i, (c, _) in enumerate(CANON)}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
NAME2CODE = {}
for c, n in CANON:
    NAME2CODE[norm(c)] = c; NAME2CODE[norm(n)] = c
NAME2CODE.update({"psalm": "PSA", "songofsongs": "SNG", "songofsolomon": "SNG", "canticles": "SNG", "revelations": "REV",
                  "1sam": "1SA", "2sam": "2SA", "1kgs": "1KI", "2kgs": "2KI", "1chr": "1CH", "2chr": "2CH",
                  "ezek": "EZK", "phil": "PHP", "jn": "JHN", "mk": "MRK", "lk": "LUK", "mt": "MAT"})

HEADING = {"s", "s1", "s2", "s3", "ms", "ms1", "mr", "r", "sr", "d", "mt", "mt1", "mt2", "mt3", "h", "toc1", "toc2",
           "toc3", "rem", "cl", "cp", "sp", "b", "restore", "ide", "sts"}
PLAIN = re.compile(r"^\s*(?:(?P<book>(?:[1-3]\s*)?[^\W\d_][^\d:\t]*?)\s*)?(?P<ch>\d+):(?P<v>\d+)[\s\t]+(?P<text>\S.*)$")


def clean(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def parse_usfm(text):
    text = re.sub(r"\\(f|fe|x)\b.*?\\\1\*", " ", text, flags=re.S)             # footnotes, cross-refs
    text = re.sub(r"\\w\s+([^|\\]*?)(?:\|[^\\]*)?\\w\*", r"\1", text)          # \w word|attrs\w*
    out, book, ch, cur = [], None, None, None
    strip = lambda s: clean(re.sub(r"\\\+?[A-Za-z0-9]+\*?", " ", s))
    for line in text.splitlines():
        m = re.match(r"\\id\s+(\w{3})", line)
        if m:
            book, ch, cur = m.group(1).upper(), None, None; continue
        m = re.match(r"\\c\s+(\d+)", line)
        if m:
            ch, cur = int(m.group(1)), None; continue
        if book is None or ch is None:
            continue
        mk = re.match(r"\\(\w+)", line)
        if mk and mk.group(1) in HEADING and "\\v " not in line:
            continue
        parts = re.split(r"\\v\s+(\d+)(?:[-,]\d+)?[a-z]?\s*", line)
        if cur is not None and strip(parts[0]):
            out[cur][3] = (out[cur][3] + " " + strip(parts[0])).strip()
        for i in range(1, len(parts), 2):
            out.append([book, ch, int(parts[i]), strip(parts[i + 1])]); cur = len(out) - 1
    return [tuple(x) for x in out if x[3]]


def parse_plain(text, default_book):
    out = []
    for line in text.splitlines():
        m = PLAIN.match(line)
        if m:
            out.append(((m.group("book") or default_book or "").strip(), int(m.group("ch")), int(m.group("v")),
                        clean(m.group("text"))))
    return out


def load_bible(paths, label):
    files = []
    for p in paths:
        files += sorted(f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if os.path.isfile(f)) \
            if os.path.isdir(p) else [p]
    entries = []
    for f in files:
        try:
            text = open(f, encoding="utf-8-sig", errors="replace").read()
        except OSError:
            continue
        if re.search(r"\\id\s+\w{3}", text[:3000]) or ("\\c " in text and "\\v " in text):
            entries += parse_usfm(text)
        else:
            entries += parse_plain(text, os.path.splitext(os.path.basename(f))[0])
    if not entries:
        sys.exit(f"Could not read any verses from the {label} Bible ({paths}). Check the format (see --help).")
    names = list(dict.fromkeys(e[0] for e in entries))
    known = all(norm(n) in NAME2CODE or n.upper() in INDEX for n in names)
    if known:
        mapping = {n: NAME2CODE.get(norm(n)) or n.upper() for n in names}
    else:                                                     # non-English book names: match by order
        start = 39 if len(names) == 27 else 0                 # 27 books => New Testament only
        if len(names) not in (66, 39, 27):
            print(f"[{label}] WARNING: {len(names)} books found (expected 66, 39 or 27). Matching by order is unreliable "
                  f"for a partial Bible. Use a USFM file, or English book names, so books can be matched by name.")
        mapping = {n: CANON[start + i][0] if start + i < len(CANON) else f"X{i}" for i, n in enumerate(names)}
        print(f"[{label}] book names not recognised; matched BY ORDER ({len(names)} books, starting at "
              f"{CANON[start][1]}). First: {names[0]} -> {mapping[names[0]]}; last: {names[-1]} -> {mapping[names[-1]]}")
    verses = {}
    for b, c, v, t in entries:
        verses.setdefault((mapping[b], c, v), t)
    print(f"[{label}] {len(files)} file(s), {len(verses)} verses, {len({k[0] for k in verses})} books")
    return verses


def backup_write(path, lines, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        shutil.copy(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n" + "".join(l + "\n" for l in lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="krio", choices=["krio", "mende", "temne"])
    ap.add_argument("--src", nargs="+", required=True, help="the local-language Bible (files/folders)")
    ap.add_argument("--english", nargs="+", required=True, help="an English Bible (files/folders)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--holdout", default="RUT,JON,PHP,JAS", help="book codes kept out for testing ('' for none)")
    ap.add_argument("--ratio", type=float, default=3.0, help="drop pairs whose length ratio exceeds this")
    ap.add_argument("--max_chars", type=int, default=600)
    a = ap.parse_args()

    src, eng = load_bible(a.src, a.lang), load_bible(a.english, "english")
    common = [k for k in src if k in eng]
    if len(common) >= 30:                                     # misalignment check: aligned verses should have similar length
        xs, ys = [len(src[k]) for k in common], [len(eng[k]) for k in common]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5 or 1
        r = cov / den
        print(f"alignment sanity: verse-length correlation {r:.2f} over {len(common)} verses")
        if r < 0.6:
            print("WARNING: low correlation. The two Bibles are probably misaligned (different versification or "
                  "wrong book matching). Inspect a few pairs in the output before training.")
    hold = {b.strip().upper() for b in a.holdout.split(",") if b.strip()}
    keys = sorted(src, key=lambda k: (INDEX.get(k[0], 999), k[1], k[2]))
    train, held, mono, dropped = [], [], [], 0
    for k in keys:
        s, e = src[k], eng.get(k)
        ok = e and 0 < len(s) <= a.max_chars and 0 < len(e) <= a.max_chars \
            and 1 / a.ratio <= len(s) / len(e) <= a.ratio
        if e and not ok:
            dropped += 1
        if ok:
            (held if k[0] in hold else train).append((s.replace("\t", " "), e.replace("\t", " ")))
        elif k[0] not in hold and 0 < len(s) <= a.max_chars:
            mono.append(s)
    if not train:
        sys.exit("No verses aligned. Do both Bibles use the same versification and book order?")
    d = a.data
    hdr = f"generated by tools/bible_align.py ({a.lang}<TAB>English)"
    backup_write(f"{d}/parallel/{a.lang}-en.tsv", [f"{s}\t{e}" for s, e in train], hdr)
    backup_write(f"{d}/heldout/{a.lang}-en.tsv", [f"{s}\t{e}" for s, e in held], hdr + " HELD-OUT BOOKS: " + ",".join(sorted(hold)))
    backup_write(f"{d}/{a.lang}.txt", mono, "verses without a usable English match")
    lens = sorted(len(s) + len(e) for s, e in train)
    p95 = lens[int(0.95 * (len(lens) - 1))]
    print(f"\naligned pairs: {len(train)} train, {len(held)} held-out ({','.join(sorted(hold)) or 'none'}); "
          f"{dropped} dropped by length filter; {len(mono)} extra monolingual lines")
    print(f"95% of pairs are <= {p95} characters together -> use about --ctx {((p95 // 3 + 12) // 32 + 1) * 32} when training")
    print("Existing files were backed up as *.bak. Next: python train.py --ctx <above> --steps <N>")


if __name__ == "__main__":
    main()
