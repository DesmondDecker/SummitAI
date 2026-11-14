"""Merge the per-source extracts in data/raw_extracts/ into the training corpus.
   python tools/build_corpus.py            (safe to re-run; rewrites data/parallel, data/heldout, data/<lang>.txt)
Every pair keeps its source in a 3rd column. Whole stories (Migeod pp. 256+) and a hash-selected slice of each other source are held out
for validation and removed from training (both sides), so evaluation scores are not inflated by repeated sentences."""
import argparse, hashlib, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean import clean_pair, english_ratio, english_vocab_from, fold_marks

STOP_END = {"the", "of", "to", "a", "an", "for", "and", "with", "in", "on", "or", "at", "by", "from", "is"}
OK_LOCAL = re.compile(r"^[A-Za-z][A-Za-z'\-\s!?,.;:]*$")
norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()
h = lambda s: int(hashlib.md5(norm(s).encode()).hexdigest(), 16)


def read(path, ncols):
    if not os.path.exists(path):
        return []
    return [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#") and len(l.split("\t")) >= ncols]


VOCAB = set()


def english_like(local):
    """Share of the 3+ letter words that are common English words (short Mende words like 'a', 'i', 'ma' are ignored)."""
    toks = [t for t in re.findall(r"[a-z']+", local.lower()) if len(t) >= 3]
    return sum(t in VOCAB for t in toks) / len(toks) if toks else 0.0


def tidy(local, en, vocab_level):
    en = re.sub(r"^[\s.,;:\-]+", "", en).strip(" ,;:")
    local = local.strip(" ,;:-")
    if vocab_level or len(local.split()) == 1:
        en, local = en.strip(" .!?"), local.strip(" .!?")
    return local, en


def good(local, en, vocab_level, lang="mende", curated=False):
    if curated == "loose":                                       # large, already-filtered sources (digits, quotes, brackets are normal)
        return bool(local.strip()) and bool(en.strip())
    if curated:                                                  # hand-checked sources: only basic sanity (identical words like bed=bed are valid)
        return bool(OK_LOCAL.match(local)) and len(local) >= 1 and len(en) >= 2
    if "ditto" in en.lower() or re.search(r"\b(see|cf)\b", en.lower()):
        return False
    toks = local.split()
    if len(toks) == 1 and len(re.sub(r"[^A-Za-z]", "", local)) < 3 and local.lower() not in {"na", "ya", "ke", "ma", "ti", "ta", "te", "wa"}:
        return False                                            # 1-2 letter 'words' are mostly OCR debris
    if lang != "krio" and len(toks) >= 2 and english_like(local) > 0.5:
        return False                                            # the 'local' side is really English / a dictionary list
    if len(toks) <= 4 and len(re.findall(r"[.!?]\s+[A-Z]", en)) >= 1 and "," in local:
        return False                                            # headword list paired with several example sentences
    if not OK_LOCAL.match(local) or local.lower() == en.lower() or len(local) < 2:
        return False
    if (local.isupper() and len(local) > 3) or (en.isupper() and len(en) > 3):      # page titles / headings in capitals
        return False
    if vocab_level:                                             # dictionary-style rows: strict
        if not re.fullmatch(r"[A-Za-z' ,/\-]+", en) or len(en.split()) > 6 or len(local.split()) > 2:
            return False
        if en.split()[-1].lower() in STOP_END or len(en) < 3:
            return False
    return True


RESTRICTED = {"jw_krio", "visitsierraleone2006", "vandevate2006"}   # copyrighted / rights unconfirmed: left out of corpora built with --release
BASE_Q = {"jw_krio": 0.8, "visitsierraleone2006": 1.0, "vandevate2006": 0.95, "mbsse_syllabus2020": 0.9, "migeod1908:phrases": 0.75, "migeod1908:stories": 0.7,
          "migeod1908:grammar_examples": 0.65, "migeod1908:topical_vocab": 0.6, "migeod1908:mende_english": 0.6, "migeod1908:english_mende": 0.6,
          "peacecorps_mende": 0.6, "peacecorps_krio": 0.6, "peacecorps_temne": 0.6}


def quality(local, en, src):
    """0-1 confidence that a pair is a correct, undamaged entry: source reliability minus penalties for OCR-damage patterns.
    Returns (q, reasons)."""
    q, why = BASE_Q.get(src, 0.5), []
    toks = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", local.lower())            # unicode letters, so words with e-open / o-open stay whole
    if any(len(t) >= 5 and not re.search(r"[aeiou\u025b\u0254]", t) for t in toks): q -= 0.3; why.append("vowelless-long-token")
    if any(re.search(r"(.)\1\1", t) for t in toks):                  q -= 0.3; why.append("triple-letter")
    if any(len(t) > 16 for t in toks):                                 q -= 0.3; why.append("very-long-token")
    if len(toks) >= 3 and sum(len(t) == 1 and t not in "aeiou" for t in toks) >= 2: q -= 0.3; why.append("spaced-letters")   # single consonants = letters split by OCR (a/e/i/o/u are real words)
    ew = en.split()
    if ew and ew[-1].lower().strip(".,") in STOP_END and len(ew) > 2:   q -= 0.2; why.append("english-truncated")
    if re.search(r"[|~^_<>{}\\]", local + en):                         q -= 0.4; why.append("stray-symbol")
    lw = max(1, len(local.split()))
    if not 0.25 <= lw / max(1, len(ew)) <= 4:                          q -= 0.2; why.append("length-mismatch")
    return max(0.0, round(q, 2)), why


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("--release", action="store_true", help="exclude sources whose rights are unconfirmed (build a corpus you can publish)")
    a = ap.parse_args()
    raw = a.data + "/raw_extracts/"
    train, held, stats = collections.defaultdict(list), collections.defaultdict(list), collections.Counter()

    cnt = collections.Counter()                               # common English words only: frequent in the English columns
    for r in read(raw + "mende_migeod.tsv", 4):
        cnt.update(t for t in re.findall(r"[a-z']+", r[3].lower()) if len(t) >= 3)
    VOCAB.update(w for w, c in cnt.items() if c >= 4)

    def add(lang, local, en, src, heldout=False, vocab=False, curated=False):
        if a.release and src in RESTRICTED:
            return
        local, en = tidy(local, en, vocab)
        if good(local, en, vocab, lang, curated):
            (held if heldout else train)[lang].append((local, en, src)); stats[(lang, src, "heldout" if heldout else "train")] += 1

    for part, bp, m, e in read(raw + "mende_migeod.tsv", 4):
        vocab = part in ("topical_vocab", "mende_english", "english_mende")
        add("mende", m, e, f"migeod1908:{part}", heldout=(part == "stories" and int(bp) >= 256 and len(m.split()) >= 3), vocab=vocab)
    for p, m, e in read(raw + "mende_manual.tsv", 3):
        add("mende", m, e, "peacecorps_mende", heldout=(h(m) % 8 == 0 and len(m.split()) >= 3))
    for p, k, e in read(raw + "krio_manual_pairs.tsv", 3):
        add("krio", k, e, "peacecorps_krio", heldout=(h(k) % 6 == 0 and len(k.split()) >= 3))
    for k, e in read(raw + "krio_vandevate.tsv", 2):
        add("krio", k, e, "vandevate2006", heldout=(h(k) % 6 == 0))
    for p, t, e in read(raw + "temne_manual.tsv", 3):
        add("temne", t, e, "peacecorps_temne", heldout=(h(t) % 7 == 0))

    for eng, kr, tm in read(raw + "vocab_visitsierraleone.tsv", 3):        # curated school-orthography vocabulary (Krio + Temne)
        for lang, loc in (("krio", kr), ("temne", tm)):
            if not loc.strip():
                continue
            locs = [x.strip() for x in loc.split("/")]; engs = [x.strip() for x in eng.split("/")]
            pairs = [(locs[0], e) for e in engs] + [(l, engs[0]) for l in locs[1:]]        # every English variant, every local variant
            for l, e in pairs:
                add(lang, fold_marks(l), e, "visitsierraleone2006", heldout=(h(locs[0]) % 6 == 0), curated=True)
    if not a.release:
        for l, e, sc in read(raw + "krio_jw.tsv", 3):               # 35k English-Krio pairs, Krio spelling kept as written
            add("krio", l, e, "jw_krio", heldout=(h(e) % 25 == 0), curated="loose")
    for m, e in read(raw + "mende_syllabus.tsv", 2):
        add("mende", fold_marks(m), e, "mbsse_syllabus2020", curated=True)

    for lang in train:                                          # dedupe, then leakage guard against the held-out sides
        seen, uniq = set(), []
        for l, e, s in train[lang]:
            k = (norm(l), norm(e))
            if k not in seen:
                seen.add(k); uniq.append((l, e, s))
        hl = {norm(l) for l, _, _ in held[lang]} | {norm(e) for _, e, _ in held[lang]}
        train[lang] = [(l, e, s) for l, e, s in uniq if norm(l) not in hl and norm(e) not in hl]
        seen = set(); hd = []
        for l, e, s in held[lang]:
            if (norm(l), norm(e)) not in seen:
                seen.add((norm(l), norm(e))); hd.append((l, e, s))
        held[lang] = hd

    os.makedirs(a.data + "/parallel", exist_ok=True); os.makedirs(a.data + "/heldout", exist_ok=True)
    for lang in ("krio", "mende", "temne"):
        for kind, d in (("parallel", train), ("heldout", held)):
            with open(f"{a.data}/{kind}/{lang}-en.tsv", "w", encoding="utf-8") as f:
                f.write(f"# {lang}\tenglish\tsource\tquality   (machine-extracted from the books in SOURCES.md; verify with native speakers)\n")
                f.writelines(f"{l}\t{e}\t{s}\t{quality(l, e, s)[0]:.2f}\n" for l, e, s in d[lang])
    flagged = [(lang, l, e, s, *quality(l, e, s)) for lang in ("mende", "krio", "temne") for l, e, s in train[lang] if quality(l, e, s)[0] < 0.5]
    with open(a.data + "/KB_REVIEW.tsv", "w", encoding="utf-8") as f:
        f.write("# lang\tlocal\tenglish\tsource\tquality\treasons   (entries the knowledge base will NOT serve until a native speaker fixes or confirms them)\n")
        f.writelines(f"{lg}\t{l}\t{e}\t{s}\t{q:.2f}\t{','.join(w)}\n" for lg, l, e, s, q, w in flagged)
    print(f"entries flagged for human review (quality < 0.5): {len(flagged)} -> {a.data}/KB_REVIEW.tsv")
    # monolingual Krio (narrations from the manual), minus anything held out
    hl = {norm(l) for l, _, _ in held["krio"]}
    mono = [r[1] for r in read(raw + "krio_manual_mono.tsv", 2) if norm(r[1]) not in hl]
    with open(a.data + "/krio.txt", "w", encoding="utf-8") as f:
        f.write("# monolingual Krio sentences (Peace Corps manual narrations, OCR; spelling normalised)\n"); f.writelines(m + "\n" for m in dict.fromkeys(mono))
    for lang in ("mende", "temne"):
        open(f"{a.data}/{lang}.txt", "w", encoding="utf-8").write(f"# no extra monolingual {lang} text yet; the {lang} side of each parallel pair is also used as monolingual text\n")
    open(a.data + "/en.txt", "w", encoding="utf-8").write("# English comes from the English side of the parallel pairs\n")
    print(f"{'language':<8}{'train pairs':>12}{'held-out':>10}   (sentence-level = 3+ words)")
    for lang in ("mende", "krio", "temne"):
        sent = sum(len(l.split()) >= 3 for l, _, _ in train[lang])
        print(f"{lang:<8}{len(train[lang]):>12}{len(held[lang]):>10}   sentence-level train: {sent}")
    print(f"monolingual Krio sentences: {len(set(mono))}")
    with open(a.data + "/SOURCES.md", "w", encoding="utf-8") as f:
        f.write("# Data sources and provenance\n\nAll pairs were machine-extracted from books you supplied (OCR / text layers) and are UNVERIFIED. "
                "Local-language text is ASCII-normalised (tone/length/dot marks and special letters removed) because OCR cannot read them reliably.\n\n"
                "| source | what | language | note on rights |\n|---|---|---|---|\n"
                "| migeod1908 | F.W.H. Migeod, *The Mende Language* (1908): phrases, grammar examples, vocabularies, stories | Mende | 1908 text, generally public domain; check your jurisdiction |\n"
                "| peacecorps_mende / _krio / _temne | Peace Corps language manuals (OCR) | Mende, Krio, Temne | US-government-produced; generally public domain, verify |\n"
                "| visitsierraleone2006 | S. U. Kamarah, *Basic Krio and Temne Vocabulary*, visitsierraleone.org (2006), school-recommended orthography; hand-curated | Krio, Temne | (c) 2006 visitsierraleone.org; confirm permission before redistributing |\n"
                "| mbsse_syllabus2020 | one example sentence from the Mende syllabus, Ministry of Basic and Senior Secondary Education (2020) | Mende | government curriculum document |\n"
                "| jw_krio | ~35k English-Krio sentence pairs from Jehovah's Witnesses publications (training_data_krio3.txt), cleaned and alignment-filtered. **Religious domain only; rights unconfirmed (publisher copyright).** | Krio | excluded by `build_corpus.py --release` |\n"
                "| vandevate2006 | M. van de Vate, *Prepositions in Krio* (interlinear examples) | Krio | copyrighted paper; short linguistic examples; check before redistributing |\n\n"
                "Considered and NOT used for training: Stanford Alpaca (CC BY-NC 4.0 research-only data, generated with an OpenAI model, so not suitable for a commercial product), "
                "the Open-Assistant GitHub repo (code only; its chat data is hosted separately and is English), programming/English textbooks and project documents "
                "(English knowledge: better served by a retrieval knowledge base than by training this model).\n\n"
                "Not used: the Berry Krio dictionary, `peace_corps_introductory_krio` and the Dwyer handbook (not yet OCR'd/extracted; see README), "
                "duplicate scans of Migeod and of the Temne manual.\n\n## Counts\n\n| language | source | split | pairs |\n|---|---|---|---|\n")
        for (lang, src, sp), n in sorted(stats.items()):
            f.write(f"| {lang} | {src} | {sp} | {n} |\n")
        if a.release:
            f.write("\n**RELEASE build.** The sources jw_krio, visitsierraleone2006 and vandevate2006 described above are NOT included in this build "
                    "(copyrighted or rights unconfirmed). The remaining sources are believed to be public-domain or government-produced "
                    "(Migeod 1908, Peace Corps manuals, a ministry syllabus) but you must verify that for your jurisdiction before publishing.\n")


if __name__ == "__main__":
    main()
