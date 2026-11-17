# KMT: Krio / Mende / Temne language system: consolidated project report

Everything built in this project, what it does, what was measured, what it cannot do, and what to do next. Numbers are from held-out tests unless stated.
The core KMT stack uses Python + NumPy. An optional locally hosted model backend and installable mobile PWA are included; model weights remain external to the repository.

## 1. Summary in five lines
* A small from-scratch transformer toolkit (autograd, BPE tokenizer, training, retrieval memory, assistant, HTTP service, simulated-robot command layer) with 47 passing tests.
* It reliably **identifies** English, Mende and Krio sentences (97-100%); Temne is weaker (76% on short items, 29% on 7 sentences).
* It **does not translate unseen text reliably** (chrF 0.03-0.29 out of 1), so it refuses to, and answers only from a quality-ranked phrasebook/dictionary memory built from your books.
* The Krio data you supplied (42k pairs) helped 2-3x but is one religious register with **unconfirmed rights**, so it is quarantined behind a `--release` switch.
* Real translation and reasoning need a pretrained model fine-tuned on a GPU; this project supplies the data pipeline and the language-specific safety/lookup layer around it.

## 2. What it can and cannot do
| Capability | Status |
|---|---|
| Language ID (krio / mende / en / temne) | works on sentences; short single words are ambiguous |
| Phrasebook + dictionary lookup (~8-76k entries), typo-tolerant, "did you mean", alternatives, quality-ranked | works; lookups of the source books, not generalisation |
| Translating new sentences | **fails** (refuses by design, calibrated on held-out data) |
| Free conversation, reasoning, coding, images | core KMT does not provide these; the optional local model backend provides chat |
| Voice input/output | the PWA supports browser-based English dictation and device read-aloud when available |
| English robot commands (move/turn/stop...) on a **simulator** | works, safety-tested; local-language commands depend on translation, so not in practice |
| Question answering from remembered facts | code exists; never trained on real data |
| HTTP API (chat/translate/identify/correct), 5 output formats, auth, rate limits, caching, correction review queue | works; ~40 req/s model-bound, ~170 req/s cache-heavy on one shared core |

## 3. The two bundles
| | `kmt_PRIVATE_full.zip` | `kmt_RELEASE_candidate.zip` |
|---|---|---|
| Code, tools, tests, docs | yes | yes |
| Corpus | everything supplied | restricted sources removed, rebuilt with `build_corpus.py --release` |
| Trained weights | `checkpoints/` (multilingual) and `checkpoints_krio/` | none (train your own: `python train.py`) |
| Safe to publish? | **No.** Contains rights-unconfirmed data and models trained on it | candidate only: verify the remaining sources' rights first (not legal advice) |

## 4. Data
| Source | Language | What | Rights note |
|---|---|---|---|
| Migeod, *The Mende Language* (1908) | Mende | phrases, grammar examples, vocabularies, stories (OCR text layer) | 1908 text; believed public domain; verify |
| Peace Corps Mende / Krio / Temne manuals | all three | dialogues, drills (OCR) | US-government-produced; believed public domain; verify |
| Ministry (MBSSE) Mende syllabus 2020 | Mende | 1 sentence | government document |
| van de Vate 2006, *Prepositions in Krio* | Krio | 54 interlinear pairs | copyrighted paper: **excluded from release** |
| visitsierraleone.org vocabulary 2006 | Krio, Temne | ~110 hand-curated words/phrases, school spelling | (c) 2006: **excluded from release** |
| `training_data_krio3.txt` | Krio | 35.5k cleaned pairs, religious domain | publisher copyright, rights unconfirmed: **excluded from release** |

Corpus sizes (train pairs / held-out pairs): **private** Mende 3,632/207, Krio 33,958/1,438, Temne 186/56; **release** Mende 3,632/207, Krio 170/21, Temne 90/22 (+978 Krio-only sentences).
Honest consequence: a release-only corpus is thin for Krio and Temne; only Mende has a few thousand pairs.
Local-language text from OCR sources is ASCII-normalised (special letters and tone marks removed); the large Krio corpus keeps its real letters. Every pair carries source and quality (0-1); 81 damaged pairs are withheld and listed in `data/KB_REVIEW.tsv` for a native speaker.
Not used: Berry's Krio dictionary and the introductory Krio manual (scans, not OCR'd), the Dwyer handbook (a bibliography), `training_data_1` (10k Nigerian-language rows, no Krio/Mende/Temne), `training_data_2..29` (~60k general English chat/code rows, unknown licences), the Alpaca zip (non-commercial licence, generated with an OpenAI model), the Open-Assistant repo (code only).

## 5. Models
| Checkpoint | Params | Trained on | Key results |
|---|---|---|---|
| `checkpoints/` multilingual | 0.72 M (d128 x3) | Mende + small Krio/Temne corpus, 2,750 steps | LID: English sentences 100%, Mende sentences 99%, Krio sentences 97%, Temne short items 76%; translation chrF 0.05-0.12 |
| `checkpoints_krio/` specialist | 0.81 M | 33.6k Krio pairs (restricted), 4,000 steps | in-domain chrF 0.21 (Krio->En), 0.29 (En->Krio); out-of-domain 0.03-0.19; held-out perplexity 170->49 and 216->66; LID 100% in-domain |
Both ship a `calibration.json` that makes the assistant abstain when held-out evidence says translations are unreliable (currently always, for unseen text). Training throughput on one shared CPU core: 13.5k tokens/s (0.16 M params) down to 445 tokens/s (11 M params).

## 6. Other measured results
* Memory lookup, typo'd 1-word / 2-word queries answered 64% -> 88% and 32% -> 94% with 0% wrong among answered; right answer offered as a suggestion ~100% when unanswered; 1.2 ms per lookup.
* Experimental word-by-word gloss: covers 38% of Mende words, 30% of glosses match; off by default.
* Robot layer: 30,000-utterance fuzz test, stop-always-wins, negation never acted on, translated commands always confirmed; each safety rule was deliberately broken to confirm a test fails.
* Bugs found and fixed along the way: float32 silently running in float64; infinite recursion on very long words; "yes" being translated instead of confirming a robot command; double-counted held-out pairs; a memory leak in the autograd graph that killed large training runs.
* Tests: 25 core + 22 service = 47 passing, including local-model routing and PWA delivery checks.

## 6b. Things that need your action
1. **Rights:** get permission for the religious Krio corpus (or drop it), and for the visitsierraleone and van de Vate material, before any public release.
2. **Security:** keep local credentials and private development data out of public releases; rotate any credentials that may have been exposed.
3. **Verification:** have native speakers check `data/KB_REVIEW.tsv` and a sample of lookups; the books were OCR'd and some entries are damaged.

## 7. Next steps (highest impact first)
1. Everyday-domain Krio (and Mende/Temne) parallel text from permissioned sources; the Bible via `tools/bible_align.py` if you hold the rights.
2. Fine-tune a pretrained open-weight multilingual model on the corpus using a GPU; keep this system as the lookup, safety and language-ID layer. OASST2 and Aya (Apache 2.0) are suitable general data.
3. Build the searchable document knowledge base for your project files (with citations) for Summit AI.
4. Real robot hardware only after a physical e-stop, firmware limits and a safety review (the layer here is simulator-only).

## 8. Quick start
```
pip install numpy
python tests/test_all.py && python tests/test_service.py
python tools/build_corpus.py            # private full corpus (or add --release)
python train.py --steps 4000            # train (add --langs krio for a specialist; --resume continues a stopped run)
python tools/calibrate.py               # decide when translations may be shown
python evaluate.py ; python tools/eval_translation_by_source.py
python chat.py                          # interactive (or: python serve.py --backend none --api-key KEY)
```
The mobile PWA is served at `/` by `serve.py`. The default backend uses a model already installed in local Ollama and does not require extra Python packages or model downloads. The model stays loaded for ten minutes after a request to speed up follow-up responses. On the development PC, a short warm response took about 1.5 seconds compared with about 14.5 seconds for a cold first request; timings vary by hardware and prompt. The Transformers backend is optional and requires a separately installed local checkpoint. Service tests cover local model routing and PWA delivery; local inference was smoke-tested through Summit's HTTP API. English voice dictation depends on browser speech support, microphone permission and HTTPS or localhost; the browser speech engine may process audio through its own service. Read-aloud uses device voices. Phone installation still needs testing on the target device over HTTPS.

## 9. File map
`kmt/` library (autograd, transformer, tokenizer, data, memory, assistant, dialogue, robot, session, formats, backends, evaluate) · `train.py finetune.py evaluate.py chat.py serve.py` ·
`tools/` (build_corpus, calibrate, eval_kb, eval_translation_by_source, extract_* and ocr_pages for the books, ingest_parallel_jsonl, inspect_chat_jsonl, bible_align, loadtest, backtranslate) ·
`data/` (parallel, heldout, raw_extracts, SOURCES.md, KB_REVIEW.tsv, robot_lexicon.tsv) · `tests/` · `README.md` (detailed notes) · `MANIFEST.sha256` (integrity list).

## 10. Not verified
Accuracy on everyday text in any of the three languages; multi-core speed; Windows/macOS; NumPy 1.x; the Anthropic backend against the real API; real robot hardware; HTTPS (terminate TLS in front); the legal status of any data (this report is not legal advice).
