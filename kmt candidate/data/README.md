# Data (all UTF-8, one item per line; lines starting with # are ignored)

| File | Format | Purpose |
|---|---|---|
| `krio.txt`, `mende.txt`, `temne.txt`, `en.txt` | one sentence per line | language modelling, language ID, generation |
| `parallel/krio-en.tsv`, `mende-en.tsv`, `temne-en.tsv` | `source<TAB>English` | translation (both directions) |
| `heldout/<lang>-en.tsv` | `source<TAB>English` | whole books kept out for honest evaluation (made by `bible_align.py`) |
| `intents.tsv` | `lang<TAB>text<TAB>label` | intent classification with YOUR labels |
| `qa.tsv` | `lang<TAB>context<TAB>question<TAB>answer` | reading comprehension (answers should be copied from the context) |
| `synthetic.tsv` | `src_lang<TAB>src<TAB>tgt_lang<TAB>tgt` | written by `tools/backtranslate.py`; training only |

Rules that matter more than model size:
* Text must come from native speakers or verified sources, never machine output.
* Choose ONE spelling convention per language (e.g. always use ɛ and ɔ) and keep it.
* Targets: ~2,000 clean lines per language to start, ~2,000 parallel pairs per language for useful translation.
* `python tools/prepare_text.py raw.txt --out data/mende.txt --append` cleans and de-duplicates raw text.
* Everything currently in `krio-en.tsv`, `krio.txt`, `en.txt`, `intents.tsv` is a small DEMO to be verified and replaced.
* Mende/Temne files are intentionally empty.
| `heldout/krio-en.tsv` | `source<TAB>English` | validation only; written by `tools/bible_align.py` (whole books kept out, so scores are honest) |

Bible input: `python tools/bible_align.py --lang krio --src <krio bible> --english <english bible>`. USFM is best. Plain text needs lines like
`Genesis 1:1 text`. If book names are not English they are matched by ORDER, so the file must be in normal Bible order, and book names must not contain digits.

## Current contents
`parallel/` and `heldout/` were built by `tools/build_corpus.py` from `raw_extracts/` (see `SOURCES.md`). Edit or add rows freely, then re-run
`python tools/build_corpus.py`, or put verified rows straight into the TSVs (the third column, `source`, is optional).
