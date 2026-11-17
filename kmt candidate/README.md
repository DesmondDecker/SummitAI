# KMT: a small AI for Krio, Mende and Temne

The KMT language toolkit is written from scratch on NumPy. Summit can use a locally hosted language model for general chat.

```
text -> BPE tokenizer -> Transformer (own autograd) -> translate / answer / classify / predict
                              ^                                   |
       retrieval memory <-----+---- corrections --> fine-tuning --+
```

## Capabilities
| | |
|---|---|
| **Understand** | language ID (krio/mende/temne/en), intent classification with your own labels, `score` (how expected is this text? flags typos/odd wording) |
| **Comprehend** | question answering from a context or from remembered facts; refuses to answer when unsure or ungrounded |
| **Predict** | next-word suggestions with probabilities (`predict_next`), text completion |
| **Translate** | each language <-> English, beam search, translation memory shortcut |
| **Remember** | instant retrieval memory of facts and corrected translations |
| **Learn** | `/fix` a translation -> stored at once -> `finetune.py` teaches the model (with replay against forgetting); `tools/backtranslate.py` creates extra training pairs from plain text |

## Summit mobile PWA and local model routing

The candidate includes a mobile-first installable web app at `/`. Chat and ask requests use the configured local Ollama model by default; translation and language identification remain on the existing KMT phrasebook/language tools. The PWA shell can load offline, but chat needs a live connection to Summit.

The default setup uses a model already installed in Ollama and does **not** download model files or add Python packages:

```powershell
ollama list
python serve.py --host 127.0.0.1 --port 8080
```

Ollama must be running locally at `http://127.0.0.1:11434`; Summit intentionally rejects remote Ollama endpoints. If the configured model is unavailable, chat returns a visible error rather than silently switching models. The local model stays loaded for ten minutes after a request to speed up follow-up responses; this uses the model's normal local memory while it is loaded. On the development PC, a short warm response took about 1.5 seconds compared with about 14.5 seconds for a cold first request; actual latency depends on the device and prompt.

An optional Transformers-based local runtime is also available for compatible checkpoints; see `python serve.py --help` for backend options. It is not needed for the default Ollama setup.

The mobile app supports English dictation and read-aloud without added packages or downloads. Dictation records audio in the browser, then sends it to Summit's local server for offline transcription through the Windows `System.Speech` engine and its installed English recognizer. The server advertises dictation availability at `/health` and accepts 16 kHz mono WAV audio at `/v1/transcribe`; audio is not sent to a cloud speech service or retained by Summit. Microphone permission and HTTPS (or localhost) are required. Choose among available input devices under Connection settings. Pause after speaking to finish dictation automatically, or tap Stop. The app reduces audio to 16 kHz mono with anti-alias filtering before upload. The transcript is inserted for review and is not sent as a chat message until you choose to send it. Read-aloud starts only when you tap a reply's button and uses an available device voice.

Summit keeps the configured local Ollama model loaded for ten minutes after a request to avoid repeated cold-start delays while allowing idle memory to be reclaimed. Chat responses default to one or two short sentences with a 96-token limit; follow-up responses can still use recent conversation context.

Open `http://127.0.0.1:8080` on the same device to test. For a phone, host Summit on a reachable machine and use HTTPS; install from Chrome's install prompt or Safari's Add to Home Screen flow. Do not expose the development server directly to the internet. Put TLS and suitable authentication in front of it. If the API key option is enabled, enter it in Connection settings; the PWA keeps it only for the current browser tab.

## Upgrades in this version, with what was actually measured
| Upgrade | Result (measured here) |
|---|---|
| KV-cache decoding (pure NumPy inference path) | **8.2x faster** generation (60 tokens, 128-d 3-layer model); output identical to the slow path (tested) |
| float32 training | **1.5x faster** steps |
| Length-bucketed batches | padding waste **40% -> 4%** |
| Beam search | implemented and tested equal to greedy at beam=1; on the toy task both scored 20/20, so **its benefit is not demonstrated here** |
| Weight averaging (EMA) + best-checkpoint saving | standard techniques; **benefit not isolated here** |
| Reading-comprehension task + retrieval | QA-only training reached **100% held-out exact match**; `ask()` answered correctly from remembered facts and abstained on an unsupported question |
| Label smoothing / dropout | **measured to hurt** context-lookup learning on small data (see below), so they are OFF by default |

### An honest finding
On the toy QA task the model learned to look answers up in the context after ~3,000 steps with no regularisation.
With label smoothing 0.05 it needed ~4,500 steps, and with dropout 0.1 as well it had barely started after 5,500.
So: leave `--dropout` and `--smoothing` at 0 until validation loss rises while training loss falls, then add them.
Also, in a single mixed multi-task run, QA did not emerge within the ~9,000 steps I could run in the sandbox even though
translation, language ID and intent had converged. Lookup skills appear late and compete with other tasks; train
longer (`--resume` lets you continue in chunks) or train QA on its own first.

## Scaling: measured numbers and honest limits
Measured on a shared **single CPU core**, float32, batch 16 (tokens/s = training throughput):

| model | parameters | tokens/s | RAM |
|---|---|---|---|
| d64  x 2 layers | 163 k | 13,500 | 0.3 GB |
| d128 x 3 layers | 720 k | 4,100 | 0.6 GB |
| d256 x 4 layers | 3.4 M | 1,300 | 1.0 GB |
| d384 x 6 layers | 11 M | 445 | 2.5 GB |

* **Data side scales fine:** 31,000 sentence pairs (125k training examples) -> tokenizer 31 s, dataset build 3 s, 223 MB.
  Retrieval memory: 60,000 items load in 6 s, fuzzy search 20 ms, exact match instant.
* **Training time is the limit.** A Bible-sized corpus is very roughly 5-10 M tokens per pass (an estimate; real verses
  are longer than my synthetic ones). At the rates above one pass takes about 0.5 h (d128) to 1.5 h (d256) to 4+ h (d384)
  on one shared core. A modern multi-core machine should be faster because NumPy's BLAS uses threads
  (set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`), but I could not test that here.
* **Practical ceiling: about 5-10 M parameters.** Beyond that, pure NumPy on a CPU is too slow; the architecture should be
  ported to PyTorch/JAX on a GPU. I did not include a port because I cannot run PyTorch here and will not ship untested code.
* Sizing advice: start `--d 128 --layers 3 --ctx 96`. If validation loss stays high while training loss is also high, grow the
  model; if training loss is low but validation loss rises, add data or `--dropout 0.1`.
* Long runs: `--resume` continues exactly (weights, optimizer, averaging), Ctrl-C saves cleanly, `--stop_after N` runs in slices,
  `--accum K` raises the effective batch without more memory. `--steps` is the TOTAL step count. After a resume the data order is reshuffled.

## What was verified, and what was not
Verified by tests or measurement (Linux, Python 3.12, NumPy 2.4): gradients vs numerical derivatives; cached decoding identical to
slow decoding; no float64 leakage in float32 mode; crash-safe atomic saves; Ctrl-C save and resume; hostile inputs (empty, emoji,
2,000-character words, 10,000-word texts, corrupted memory file, mismatched tokenizer); Bible alignment (USFM and plain text, BOM,
CRLF, footnotes, headings, whole-book hold-out with no leakage); 60k-item memory; every command-line tool end to end.
Two of the regression tests were mutation-checked (the bug was reintroduced and the test failed).

**Not verified:** the Anthropic backend against the real API; any real robot hardware; HTTPS (terminate TLS in front); accuracy on real Mende/Temne/Krio (no real data here); multi-core speed; Windows/macOS; NumPy 1.x; any benefit
of beam search, EMA or back-translation; a GPU/PyTorch port. "Works perfectly" cannot honestly be promised for real data until you run it on yours.

## Your uploaded books -> training data -> what the model can actually do (measured)
Extracted with `tools/` (coordinate-based two-column pairing, OCR for scans, strict filters). Every pair keeps its source in `data/parallel/*.tsv`; raw per-source
extracts are in `data/raw_extracts/`; counts and rights notes are in `data/SOURCES.md`.

| language | training pairs | held out (never trained on) | other text |
|---|---|---|---|
| Mende | 3,632 (1,889 sentences) | 207 (whole stories) | |
| Krio | 329 (187 sentences) | 52 | 978 Krio-only sentences |
| Temne | 186 (16 sentences, the rest words/short phrases) | 56 | |

Sources used: Migeod 1908 (Mende), Peace Corps Mende / Krio / Temne manuals, van de Vate 2006 (Krio), and the hand-curated *Basic Krio and Temne Vocabulary*
(visitsierraleone.org, 2006, school-recommended spelling: the cleanest source here and the one that doubled the Temne data). One Mende example sentence from the 2020 syllabus.
**Considered and not used for training:** Stanford Alpaca (research-only CC BY-NC data generated with an OpenAI model: not suitable for a commercial product),
the Open-Assistant GitHub repo (code only; its chat data is hosted elsewhere and is English), and the English textbooks / project documents (better as a retrieval
knowledge base than as training text for this model). **Not yet OCR'd:** Berry's Krio dictionary, `peace_corps_introductory_krio`; the Dwyer handbook is a bibliography.
Local-language text is ASCII-normalised (special letters and tone marks removed; ɛ->e, ɔ->o, ŋ->ng, ʌ and ə->a), and the tokenizer applies the same fold to typed input.

**Trained model** (0.72 M parameters, d128 x 3 layers, dropout 0.1, best held-out checkpoint at step 2,750; `checkpoints/`). Language-ID accuracy on held-out text,
old model (earlier corpus) vs this one, scored on the SAME held-out set:

| held-out text | n | old | new |
|---|---|---|---|
| English sentences | 243 | 99% | 100% |
| Mende sentences | 207 | 100% | 99% |
| Krio sentences | 29 | 100% | 97% |
| Krio single words / short phrases | 23 | 26% | 35% |
| Temne words / short phrases | 49 | 53% | **76%** |
| Temne sentences | 7 | 29% | 29% (too few to conclude) |

Reading it plainly: **sentences in English, Mende and Krio are identified reliably; single words are not** (many Krio words are identical to English words, so a one-word
guess is inherently ambiguous), and the extra Temne vocabulary clearly helped short Temne items. **Translation still does not work**: held-out chrF is 0.05-0.12 in every
direction (it writes fluent text that ignores the input), and `tools/calibrate.py` therefore keeps the assistant refusing to translate unseen text. Known
phrasebook/dictionary entries (about 8,300 in the translation memory) still answer instantly, but those are lookups of the books themselves, not generalisation, and some
are OCR-damaged: spot-check them with native speakers.

**Knowledge base (the translation memory) was audited and optimised.** Measured with `tools/eval_kb.py` on typo'd copies of stored entries (simulated random typos, not real user
typing) and on held-out text the memory has never stored. At the real accept threshold the old policy (accept only a >= 0.92 character-similarity match) answered just
64% / 32% / 34% of typo'd 1-word / 2-word / 3+-word queries and never offered a suggestion (an earlier "94.8%" figure ignored that threshold, so do not compare it).
Now: keys ignore case, punctuation and special letters, so duplicates merge and alternatives are kept; answers are ranked by a per-entry quality score
(source reliability minus OCR-damage penalties); an edit-distance check handles short words; a short word is auto-answered only if exactly one entry is within 1-2 edits
(ambiguous cases become "did you mean ..." suggestions); every near match tells the user which entry was matched.

| typo'd query | answered (old -> new) | wrong among answered | right answer offered as a suggestion when not answered |
|---|---|---|---|
| 1 word | 64% -> 88% | 0% | 100% (old: never) |
| 2 words | 32% -> 94% | 0% | 93% |
| 3+ words | 34% -> 37% | 0% | 100% |

On 144 unseen short items the new lookup answered none (so no wrong answers); on 486 unseen sentences it accepted 4 near-matches, 3 of them harmless variants
("running to" vs "running towards", "She said" vs "He said", where Mende has no gender) and 1 genuine error. 81 pairs (162 directional entries, about 2% of the memory) fall
below quality 0.5 and are withheld until a native speaker fixes them: they are listed in `data/KB_REVIEW.tsv`. Lookup takes about 1.2 ms over 8,132 entries.
After rebuilding the corpus, delete `memory/memory.jsonl` so the memory is re-seeded. These are lookups of the books you supplied, not translation ability.

Experimental `gloss()` / `/gloss`: word-by-word dictionary lookup. On held-out Mende sentences it covers only 38% of words and 30% of those glosses appear in the reference
translation, and Krio has too few dictionary entries to gloss anything, so it is NOT shown by default.

What would actually give usable translation: (1) far more clean parallel data (the Bible via `tools/bible_align.py`: about 30k aligned verses), (2) fine-tuning a pretrained
multilingual model on this corpus instead of training from zero (needs PyTorch and a GPU; not possible in the sandbox used to build this), (3) native-speaker verification.

## Large Krio corpus (42k pairs) and what it did (measured)
`training_data_krio3.txt` held 42,248 English-Krio pairs. `tools/ingest_parallel_jsonl.py` filtered them (length ratio, mismatched numbers, citation fragments, duplicates,
a cap on formulaic "(Read ...)" lines) and scored alignment with IBM Model 1 learned from the data itself, dropping the worst 8%: 35,548 pairs kept, of which 33,629 train and
1,386 are held out (plus 52 held-out pairs from the other Krio sources). **Caveats:** the text is almost entirely religious-domain, from Jehovah's Witnesses publications, and its
**rights are unconfirmed** (publisher copyright): it is marked restricted, `python tools/build_corpus.py --release` leaves it out, and weights trained on it should not be published
(see `checkpoints_krio/PROVENANCE.md`; not legal advice). Krio spelling (e-open, o-open, eng) is kept as written for this source.

Krio specialist (`checkpoints_krio/`, 0.81 M parameters, 4,000 steps), held-out, spelling-insensitive chrF (0-1) from `tools/eval_translation_by_source.py`:

| held-out set | Krio -> English | English -> Krio |
|---|---|---|
| in-domain (same religious style as training) | 0.21 | 0.29 |
| out-of-domain (phrasebook, vocabulary, linguistics sentences) | 0.03 - 0.19 | 0.03 - 0.19 |
| for comparison: earlier model on the earlier Krio data | about 0.09 | about 0.09 |

Held-out perplexity fell from 170 to 49 (Krio->English) and from 216 to 66 (English->Krio). So 100x more Krio parallel data helped 2-3x, but **it still does not translate reliably**:
outputs read as English yet often say something else, and it is weak outside the religious domain. `tools/calibrate.py` therefore leaves the assistant refusing to translate unseen text.
Realistic next steps: fine-tune a pretrained multilingual model on this corpus (GPU), and add everyday-domain Krio parallel text (conversation, news, health, agriculture) so it does not only know one register.

**Bug found and fixed while training this:** the autograd graph held reference cycles that Python freed only at occasional full garbage collections, so memory grew about 10 MB per step on
large datasets until the process was killed at ~4 GB. Backward now frees the graph immediately (memory stays at ~300 MB); `test_graph_memory_is_freed_after_backward` guards it.

**Other uploads (inventory only, not used):** `training_data_1.txt` is 10,000 rows of "Hypa Translate" data for Nigerian languages (Urhobo, Igbo, Tiv, Efik, Hausa and others) plus dictionary and
language-detection rows, with LLM-style reasoning traces; it has no Krio, Mende or Temne text. `training_data_2` to `_29` are roughly 60k rows of general English chat, instruction, function-calling and Python-code
data. Their licences and origins are unknown, so none were used; they would matter only for a later GPU fine-tune of a general model, after a licence check (`tools/inspect_chat_jsonl.py` audits such files).

## Serving, context, formats and the robot layer (new)
```
python serve.py --host 127.0.0.1 --port 8080                                 # local Ollama model is the default
python serve.py --backend none --host 127.0.0.1 --port 8080                 # API/UI work without a chat model
python serve.py --help                                                       # see local backend options
python serve.py ... --robot sim --api-key SECRET                             # simulated robot; REQUIRES --api-key
python tools/loadtest.py --port 8080 --clients 8 --requests 100
```
Endpoints: PWA `GET /`; `POST /v1/chat` (context-aware routing), `/v1/translate`, `/v1/identify`, `/v1/correct`, `/v1/robot/stop`, `GET /health`, `/metrics`.
Every reply can be `json`, `text`, `markdown`, `xml` or `csv` (`?format=`); Markdown/XML output is escaped so replies cannot inject markup.

**Measured service speed** (one shared CPU core, 8 concurrent clients, plain stdlib HTTP server, ~0.7 M-parameter model):
about **40 requests/s** when the model runs every time (p50 136 ms), **~80/s** with 50% repeated phrases, **~170/s** when most phrases repeat (cache).
Scale by running one process per core behind a load balancer; the server is stateless apart from bounded in-memory sessions.
Planning estimate (assumptions, not a measurement): 1 M users x 10% active daily x 10 requests is ~12 requests/s on average and
perhaps 100/s at peak, which a few cores handle. A faster HTTP front end (uvicorn/gunicorn) would raise the per-core ceiling; I did not test one.
I measured a fused/preallocated-cache decoder at only ~1.2x, so it was not added.

**Built for many users:** per-client rate limits, bounded concurrency (HTTP 503 + Retry-After when busy), request size limits, optional API key
(constant-time compare), no user text in logs, bounded expiring sessions, LRU cache, and **corrections are quarantined for human review by
default** (`--accept-corrections` applies them at once; leave it off for a public service, otherwise anyone can poison translations for everyone).

**"Context aware" means:** per-session language, remembered facts, the last translation (so "and in Mende?" works), pending confirmations, and
recent history for the optional large-model backend. It does not give the small model deep reasoning or long-range memory.

**The configured chat model runs locally on the server.** Ordinary chat and questions use the local model; it does not make robot decisions. Explicit translation and identification tasks continue through KMT. The optional Anthropic backend remains available only when selected and configured; it sends user text to a third party.

### Operating a robot: what this is and is not
The language model **never produces motor commands**. It may only translate. A small deterministic parser turns English text into one of a few
allow-listed commands (move, turn, speed, pick up, put down, status, stop), a validator enforces hard limits (distance 0.05-5 m, turn 1-360 degrees,
speed 0.05-0.5 m/s), and these safety rules are tested:
* **stop always wins**: any stop word, in any sentence, preempts everything, skips rate limits, clears pending commands, and has its own endpoint.
* **negation is never acted on** ("do not go forward" asks for a clear command); multi-action sentences and missing distances ask a question instead of guessing.
* **commands translated from Krio/Mende/Temne are always echoed and need an explicit yes**, and low-confidence translations are refused.
* risky commands (moves of 2 m or more, pick up/put down) need confirmation; confirmations expire after 30 s; rate limit of 10 commands per 10 s.
* the adapter re-checks limits and a geofence independently; a command that would leave the area is refused, not clipped.
* robot control will not start without an API key; everything is audit-logged.
The safety tests include a 30,000-utterance fuzz test and deliberately broken versions of each safety feature (all detected).
**Only a simulator is included.** To drive real hardware you implement `RobotAdapter` yourself. A language layer is not a safety system:
a real robot also needs a physical emergency stop, firmware-level speed/torque/bump limits, and a risk assessment by a qualified engineer
before anyone stands near it. Local-language command words in `data/robot_lexicon.tsv` (including `stɔp`) are UNVERIFIED and must be checked
by native speakers; English is the only fully supported command language today. It cannot do perception, navigation or planning in the real world.

## Layout
```
kmt/autograd.py     autodiff: matmul, softmax, layer-norm, GELU, embeddings, cross-entropy, float32/64
kmt/transformer.py  transformer; training forward + fast cached inference, beam search, scoring
kmt/tokenizer.py    BPE tokenizer trained on YOUR text        kmt/optim.py   AdamW + EMA
kmt/data.py         tasks (lm, translate, intent, qa), balancing, bucketing, feedback log, synthetic pairs
kmt/memory.py       retrieval memory                          kmt/assistant.py  everything together
kmt/evaluate.py     perplexity, language ID, chrF, QA accuracy, next-token accuracy
train.py  finetune.py  evaluate.py  chat.py
tools/prepare_text.py  tools/make_toy_data.py  tools/backtranslate.py  tools/build_corpus.py  tools/calibrate.py
tools/twocol.py  tools/clean.py  tools/ocr_pages.py  tools/extract_migeod.py  tools/extract_krio_manual.py  tools/extract_peacecorps_dialogues.py  tools/extract_interlinear.py
tests/test_all.py + tests/test_service.py   47 tests: service, robot fuzzing, dialogue, gradients, causality, cached==uncached decoding, float32, robustness, memory, Bible tool
tools/bible_align.py  verse-align a local-language Bible with an English one (USFM or plain text), holds out whole books
```

## Quick start
```
pip install numpy
python tests/test_all.py && python tests/test_service.py
python tools/make_toy_data.py --out toy_data          # synthetic languages, to prove the pipeline learns
python train.py --data toy_data --out toy_ckpt --steps 2500 --d 96 --layers 3 --ctx 64 --merges 150 --lr 3e-3
python evaluate.py --data toy_data --ckpt toy_ckpt
```
Real data (see `data/README.md`): `python train.py --steps 6000`, then `python evaluate.py`, then `python chat.py`.
Chat commands: plain text (auto-translate), `/tr krio>en ...`, `/ask ...`, `/next ...`, `/score ...`, `/intent ...`,
`/id ...`, `/gen krio ...`, `/remember ...`, `/fix <correct translation>`.

The mobile PWA and local model server use the separate setup above; the NumPy-only quick start does not install model runtimes or weights.

## What this cannot be
It is a small specialist, not a general assistant. Its ceiling is set by your data: thousands of clean native-speaker lines
per language for language ID and Krio, and real parallel pairs for Mende/Temne translation. It will not have the world
knowledge or reasoning of a large model. Toy results (~100%) come from tiny, regular synthetic languages and say nothing
about real-language accuracy. Back-translation is unverified here and can amplify errors; spot-check its output.
