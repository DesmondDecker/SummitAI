"""The assistant: model + tokenizer + retrieval memory + feedback loop."""
import json, os, re
import numpy as np
from .transformer import Transformer
from .tokenizer import BPE
from .memory import MemoryStore
from .data import (LANGS, tag, mono_prompt, translate_prompt, intent_prompt, qa_prompt, read_tsv,
                   feedback_append)
from .tokenizer import SP


def _softmax(x):
    x = np.asarray(x, dtype=float); e = np.exp(x - x.max()); return e / e.sum()


class Assistant:
    def __init__(self, ckpt_dir="checkpoints", memory_dir="memory", data_dir="data"):
        self.ckpt = os.path.join(ckpt_dir, "kmt.npz")
        tok_path = os.path.join(ckpt_dir, "tokenizer.json")
        for path in (self.ckpt, tok_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found. Train first:  python train.py  (see README)")
        self.model, self.extra = Transformer.load(self.ckpt)
        self.tok = BPE.load(tok_path)
        if len(self.tok) != self.model.cfg["vocab_size"]:
            raise ValueError(f"tokenizer has {len(self.tok)} tokens but the model expects {self.model.cfg['vocab_size']}: "
                             "they come from different training runs. Use matching files from the same --out folder.")
        self.ctx = self.model.cfg["ctx"]
        self.langs = self.extra.get("langs") or LANGS
        self.intents = self.extra.get("intents", [])
        cal = os.path.join(ckpt_dir, "calibration.json")             # written by tools/calibrate.py from held-out evidence
        self.calibration = json.load(open(cal)) if os.path.exists(cal) else {}
        self.feedback_path = os.path.join(memory_dir, "feedback.jsonl")
        self.memory = MemoryStore(os.path.join(memory_dir, "memory.jsonl"))
        self.ban = [self.tok.pad, self.tok.unk, self.tok.bos, self.tok.sep] + \
                   [self.tok.id(t) for t in ("<tr>", "<intent>", "<qa>") + tuple(tag(l) for l in LANGS)]
        if len(self.memory) == 0:                         # seed translation memory from your parallel data
            self._seed(data_dir)

    def _seed(self, data_dir, qmin=0.5):
        """Fill the translation memory from data/parallel/*.tsv (columns: local, English, source, quality).
        Entries below quality `qmin` are left out (they are listed in data/KB_REVIEW.tsv for a native speaker to fix)."""
        for lang in ("krio", "mende", "temne"):
            path = os.path.join(data_dir, "parallel", f"{lang}-en.tsv")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    p = line.rstrip("\n").split("\t")
                    if len(p) < 2 or not p[0].strip() or not p[1].strip():
                        continue
                    q = float(p[3]) if len(p) > 3 and re.fullmatch(r"[0-9.]+", p[3]) else 1.0
                    if q >= qmin:
                        self.memory.add_pair(lang, p[0].strip(), "en", p[1].strip(), q=q)

    # -------------------------------------------------------- understanding
    def identify(self, text):
        """Which language makes this text most probable? Returns (best, {lang: prob})."""
        ids = self.tok.encode(text or "")[: self.ctx - 4]
        batch = np.array([mono_prompt(self.tok, l) + ids + [self.tok.eos] for l in self.langs])
        lp = self.model.seq_logprob(batch, start=2)
        pr = _softmax(lp)
        probs = {l: float(p) for l, p in zip(self.langs, pr)}
        return max(probs, key=probs.get), probs

    def classify_intent(self, text, lang=None):
        if not self.intents:
            return None, {}
        lang = lang or self.identify(text)[0]
        prompt = intent_prompt(self.tok, lang, text)[: self.ctx - 12]
        lps = [float(self.model.seq_logprob(np.array([prompt + self.tok.encode(lab) + [self.tok.eos]]),
                                            start=len(prompt))[0]) for lab in self.intents]
        pr = _softmax(lps)
        probs = dict(zip(self.intents, map(float, pr)))
        return max(probs, key=probs.get), probs

    # -------------------------------------------------------- generation
    def _split(self, text, budget):
        """Split long text into pieces of at most `budget` tokens: by sentence, then by words."""
        pieces, cur, cur_n = [], [], 0
        for sent in re.split(r"(?<=[.!?;:])\s+", text.strip()):
            units = [sent] if len(self.tok.encode(sent)) <= budget else sent.split()
            units = [u2 for u in units for u2 in                    # a single giant "word": cut by characters
                     ([u] if len(self.tok.encode(u)) <= budget else [u[i:i + budget] for i in range(0, len(u), budget)])]
            for unit in units:
                n = len(self.tok.encode(unit))
                if cur and cur_n + n > budget:
                    pieces.append(" ".join(cur)); cur, cur_n = [], 0
                cur.append(unit); cur_n += n
        if cur:
            pieces.append(" ".join(cur))
        return pieces

    def translate(self, text, src_lang=None, tgt_lang="en", use_memory=True, mem_threshold=0.92, beam=4, _nested=False, force=False, short_auto=True):
        text = " ".join((text or "").split())
        if not text:
            return dict(text="", source="empty", src_lang=src_lang, tgt_lang=tgt_lang)
        src_lang = src_lang or self.identify(text)[0]
        budget = (self.ctx - 8) // 2 - 6                     # source tokens that leave room for the answer
        if len(self.tok.encode(text)) > budget and not _nested:   # long input: translate piece by piece
            parts = [self.translate(p, src_lang, tgt_lang, use_memory, mem_threshold, beam, _nested=True, force=force) for p in self._split(text, budget)]
            res = dict(text=" ".join(p["text"] for p in parts if p["text"]), src_lang=src_lang, tgt_lang=tgt_lang,
                       source="model" if any(p["source"] == "model" for p in parts) else parts[0]["source"], chunks=len(parts))
            confs = [p["confidence"] for p in parts if "confidence" in p]
            if confs:
                res["confidence"] = min(confs)
            warn = [p["warning"] for p in parts if "warning" in p]
            if warn:
                res["warning"] = warn[0]
            return res
        if src_lang == tgt_lang:
            return dict(text=text, source="unchanged", src_lang=src_lang, tgt_lang=tgt_lang)
        suggestions = []
        if use_memory:
            lk = self.memory.lookup(text, src_lang, tgt_lang, accept=mem_threshold, short_auto=short_auto)
            if lk["status"] in ("exact", "near"):
                return dict(text=lk["answer"], source="memory", score=lk["score"], match=lk["status"], matched=lk["matched"],
                            alternatives=lk["alternatives"], src_lang=src_lang, tgt_lang=tgt_lang)
            suggestions = lk["suggestions"]
        prompt = translate_prompt(self.tok, src_lang, text, tgt_lang)
        if len(prompt) > self.ctx - 8:                       # cannot happen after splitting; never truncate silently
            return dict(text="", source="error", error="input too long", src_lang=src_lang, tgt_lang=tgt_lang)
        n_src = len(self.tok.encode(text))
        if beam > 1:
            out, conf = self.model.beam_search(prompt, 2 * n_src + 12, self.tok.eos, beam, self.ban)
        else:
            out, conf = self.model.generate(prompt, 2 * n_src + 12, self.tok.eos, self.ban)
        res = dict(text=self.tok.decode(out), source="model", confidence=float(np.exp(conf)),
                   src_lang=src_lang, tgt_lang=tgt_lang)
        thr = self.calibration.get("directions", {}).get(f"{src_lang}>{tgt_lang}", {}).get("min_conf", self.calibration.get("default", 0.0))
        if not force and res["confidence"] < thr:                 # held-out evidence says this is not reliable: refuse, offer known examples
            if not suggestions:
                suggestions = self.memory.lookup(text, src_lang, tgt_lang)["suggestions"]
            return dict(text="", source="abstained", confidence=res["confidence"], src_lang=src_lang, tgt_lang=tgt_lang,
                        reason="not reliable enough yet (measured on held-out data)", suggestions=suggestions)
        for l in (src_lang, tgt_lang):
            if l not in self.langs:
                res["warning"] = f"the model has no training data for '{l}' yet"
        return res

    def generate(self, lang, prompt="", max_new=60, temperature=0.8, top_k=20, seed=None):
        prefix = (mono_prompt(self.tok, lang) + self.tok.encode(prompt))[: self.ctx - 2]
        out, _ = self.model.generate(prefix, max_new, self.tok.eos, self.ban, temperature, top_k, seed)
        return (prompt + " " + self.tok.decode(out)).strip() if prompt else self.tok.decode(out)

    # -------------------------------------------------------- reading aid
    def _lexicon(self):
        """word -> English gloss (most frequent first), from single-word translation-memory entries."""
        if getattr(self, "_lex", None) is None:
            from collections import Counter, defaultdict
            c = defaultdict(Counter)
            for it in self.memory.items:
                if it["kind"] == "translation" and it.get("tgt_lang") == "en" and len(it["src"].split()) == 1:
                    c[(it["lang"], self.tok.norm(it["src"]).lower().strip(".,!?;:"))][it["tgt"].strip().lower()] += 1
            self._lex = {k: v.most_common(1)[0][0] for k, v in c.items()}
        return self._lex

    def gloss(self, text, lang):
        """Word-by-word glosses from the dictionary entries. A rough READING AID, not a translation:
        word order, grammar and unknown words are not handled. Returns {'words': [(word, gloss|None)], 'coverage': share found}."""
        lex = self._lexicon()
        words = re.findall(r"[^\W\d_]+(?:['\-][^\W\d_]+)*", self.tok.norm(text or ""))
        out = [(w, lex.get((lang, w.lower()))) for w in words]
        found = sum(g is not None for _, g in out)
        return dict(words=out, coverage=(found / len(out)) if out else 0.0)

    # -------------------------------------------------------- comprehension & prediction
    def ask(self, question, lang=None, context=None, k=2, min_conf=0.6, extractive=True):
        """Answer a question from a given context, or from the top-k most relevant remembered facts.
        Abstains (answer=None) when confidence is low or, if extractive=True, when the answer
        does not literally appear in the context (guards against made-up answers)."""
        lang = lang or self.identify(question)[0]
        support = []
        if context is None:
            support = [it["text"] for _, it in self.recall(question, k)]
            context = " ".join(support)
        if not context:
            return dict(answer=None, reason="no context or remembered facts", support=[])
        prompt = qa_prompt(self.tok, lang, context, question)
        if len(prompt) > self.ctx - 8:
            return dict(answer=None, reason="context too long", support=support)
        out, conf = self.model.beam_search(prompt, 24, self.tok.eos, 3, self.ban)
        c = float(np.exp(conf)); ans = self.tok.decode(out)
        grounded = (not extractive) or ans.lower() in context.lower()
        ok = bool(ans) and c >= min_conf and grounded
        return dict(answer=ans if ok else None, guess=ans, confidence=c, grounded=grounded,
                    support=support, context=context)

    def predict_next(self, text, lang="en", k=5):
        """Top-k most likely next tokens (predictive text). Returns [(token, probability)]."""
        prefix = mono_prompt(self.tok, lang) + self.tok.encode(text)
        lp = self.model.next_token_dist(prefix[-(self.ctx - 1):], self.ban + [self.tok.eos])
        return [(self.tok.itos[i].replace(SP, " "), float(np.exp(lp[i]))) for i in np.argsort(-lp)[:k]]

    def score(self, text, lang=None):
        """How expected is this text? Low-probability tokens often reveal typos or unusual wording."""
        lang = lang or self.identify(text)[0]
        ids = mono_prompt(self.tok, lang) + self.tok.encode(text)[: self.ctx - 4] + [self.tok.eos]
        lps = self.model.token_logprobs(ids)[1:]                 # skip the language-tag prediction
        toks = [self.tok.itos[i].replace(SP, " ") for i in ids[2:]]
        return dict(lang=lang, avg_logprob=float(lps.mean()), perplexity=float(np.exp(-lps.mean())),
                    tokens=list(zip(toks, map(float, np.exp(lps)))))

    # -------------------------------------------------------- memory & learning
    def remember(self, text, lang=None):
        return self.memory.add_fact(text, lang)

    def recall(self, query, k=3):
        return self.memory.search(query, k, kind="fact")

    def correct_translation(self, src_lang, src, tgt_lang, tgt):
        """A human correction: instantly stored in memory, queued for the next fine-tune."""
        self.memory.add_pair(src_lang, src, tgt_lang, tgt)
        feedback_append(self.feedback_path, dict(task="translate", src_lang=src_lang, src=src,
                                                 tgt_lang=tgt_lang, tgt=tgt))

    def correct_intent(self, lang, text, label):
        feedback_append(self.feedback_path, dict(task="intent", lang=lang, text=text, label=label))

    def add_text(self, lang, text):
        feedback_append(self.feedback_path, dict(task="lm", lang=lang, text=text))
