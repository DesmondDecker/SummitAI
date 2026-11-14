"""Shared text cleaning for pairs extracted from OCR'd language books."""
import re, unicodedata

_PAREN = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_WH = re.compile(r"^(what|where|when|why|how|which|who|whom|do|does|did|is|are|am|was|were|have|has|can|will|shall|would|could|may)\b", re.I)
LOCAL_OK = re.compile(r"^[A-Za-z][A-Za-z'’\-\s!?,.;:]*$")


def fold_marks(s):
    """Strip tone/length/dot-below marks (OCR cannot read them reliably) and map open vowels to plain letters."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.translate(str.maketrans({"ɛ": "e", "Ɛ": "E", "ɔ": "o", "Ɔ": "O", "ŋ": "ng", "Ŋ": "Ng", "ɓ": "b", "ɗ": "d", "ʌ": "a", "ə": "a", "ε": "e", "ʌ": "a", "ə": "e", "’": "'", "‘": "'"}))


def tidy_punct(s):
    s = re.sub(r"\s+([!?,.;:])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -—–_*•·")
    return s


def fix_question_mark(en):
    """OCR often reads '?' as 1, ], ) or 2 at the end of a question."""
    if _WH.match(en.strip()):
        en = re.sub(r"\s*[1\]2)lI|]+\s*\.?$", "?", en.strip())
    return en


_SINGLE_JUNK = set("bcdfghjklmnpqrstvwxyz")


def spaced_letter_noise(local):
    """OCR sometimes spaces letters inside words ('y a', 'n gi'); such lines cannot be repaired reliably."""
    toks = re.findall(r"[A-Za-z']+", local.lower())
    return any(len(t) == 1 and t in _SINGLE_JUNK for t in toks)


def clean_pair(left, right, local_first_variant=True):
    """Returns (local, english) or None if the pair looks unusable."""
    right = re.sub(r"(\w)- (\w)", r"\1\2", right)          # 'wo- man' -> 'woman' (hyphenation at line breaks)
    left = re.sub(r"(\w)- (\w)", r"\1\2", left)
    right = _PAREN.sub(" ", right)                       # '(pl.)', '(lit. ...)': notes, not translation
    left = _PAREN.sub(" ", left)
    if local_first_variant:
        left = re.split(r"\s+or\s+|,\s*or\s+|\s*/\s*", left)[0]      # 'Li la, or Di la !' -> first variant
    left, right = tidy_punct(fix_question_mark(left)), tidy_punct(fix_question_mark(right))
    right = fix_question_mark(right)
    left = fold_marks(left)
    if not (1 <= len(left) <= 90 and 2 <= len(right) <= 180):
        return None
    if not LOCAL_OK.match(left) or not re.search(r"[A-Za-z]{2}", right):
        return None
    if sum(c.isascii() for c in right) / len(right) < 0.97:
        return None
    if re.search(r"\d", left) or len(re.findall(r"\d", right)) > 2:
        return None
    if spaced_letter_noise(left) or re.search(r"\.\.|\b[a-z]\s[a-z]\s[a-z]\b", right):
        return None
    if not (0.15 <= len(left) / max(1, len(right)) <= 6):
        return None
    if re.search(r"\b(adj|pron|prep|conj|interj|tr|intr|adv|suffix|plur)\b\.?", right) and len(right.split()) < 4:
        return None
    return left, right


# ---------------------------------------------------------------- language-ness heuristics
ENGLISH_STOP = set("the of and is are to in that this which with for as by have has been was they their from an it be on at or we you your what who "
                   "when where how not do does did will would can could should there here these those them his her its our but if so than then "
                   "also very much many some any all each one two three".split())
KRIO_MARKERS = set("de dehn dem na foh fo noh wetin udat uskain uspat aw di yu mi wi una dohn kin bin mek pikin se ehn en i in dohn go gi tehl tel "
                   "bohku bohku lilibit tumara tide naw usai wanwan padi pas".split())


def pc_krio_to_std(s):
    """Peace Corps Krio spelling -> plain ASCII standard: 'eh'->'e', 'oh'->'o' inside words (tehl->tel, Gohd->God, foh->fo)."""
    def fix(m):
        w = m.group(0)
        return re.sub(r"eh", "e", re.sub(r"oh", "o", w, flags=re.I), flags=re.I) if re.search(r"[eo]h", w, re.I) else w
    return re.sub(r"[A-Za-z']+", fix, s)


def english_ratio(text, vocab):
    toks = re.findall(r"[a-z']+", text.lower())
    return (sum(t in vocab or t in ENGLISH_STOP for t in toks) / len(toks)) if toks else 0.0


def stop_hits(text):
    return sum(t in ENGLISH_STOP for t in re.findall(r"[a-z']+", text.lower()))


def krio_like(text):
    toks = re.findall(r"[a-z']+", text.lower())
    return sum(t in KRIO_MARKERS for t in toks) >= 1 and stop_hits(text) <= max(1, len(toks) // 4)


def english_vocab_from(tsv):
    """English word set from the English column (last field) of a TSV of extracted pairs."""
    v = set()
    for l in open(tsv, encoding="utf-8"):
        if not l.startswith("#"):
            v |= set(re.findall(r"[a-z']+", l.rstrip("\n").split("\t")[-1].lower()))
    return v
