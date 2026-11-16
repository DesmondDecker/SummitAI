# Summit Qwen Migration Record

## Decision

The legacy NumPy model cannot accept Qwen parameters: its transformer blocks, positional embeddings, tokenizer, vocabulary, and checkpoint format are different. The migration therefore adds a parallel Qwen3 path and keeps the existing Summit stack available.

Selected foundation: `Qwen/Qwen3-0.6B-Base`, official model card lists 0.6B parameters and Apache-2.0. This is the size selected for local feasibility; 1.7B and 4B are larger alternatives. Qwen3 Base is pretrained-only, so instruction following and conversational behavior require supervised fine-tuning.

## Preserved

- Existing `checkpoints/` and `checkpoints_krio/` remain unchanged and continue to load with the NumPy assistant.
- Translation memory, language-ID and phrasebook logic, session manager, dialogue router, HTTP API, format handlers, and deterministic robot safety remain in place.
- Qwen is opt-in via `chat_qwen.py` or `serve.py --backend qwen`; prompts are not sent to a remote service.

## Added

- `kmt/qwen_runtime.py`: local Transformers inference, Qwen chat-template use, bounded conversation trimming, and optional adapter loading.
- `chat_qwen.py`: local conversational CLI with persisted recent turns and `/reset`.
- `tools/prepare_qwen_data.py`: OASST2/Aya English filtering, OASST conversation reconstruction, source manifest, local translation-source allowlist, and grouped splits.
- `tools/train_qwen_lora.py`: response-only LoRA SFT with bounded CPU-friendly defaults, validation loss logging, and adapter-only output.
- `tools/evaluate_qwen.py`: held-out conversation overlap and translation chrF; coding and cultural metrics are explicitly marked unavailable until benchmark data exists.
- `serve.py --backend qwen`: optional local Qwen conversational backend while preserving the original API and safety routing.

## Data and License Notes

- Qwen3 model card: Apache-2.0; retain the official model card and license with any redistributed derivative.
- OASST2 and Aya dataset cards identify Apache-2.0. Exact revisions are recorded in `data_qwen/manifest.json` when the data-preparation step succeeds.
- Local translation pairs are limited to Peace Corps, Migeod 1908, and MBSSE syllabus sources with quality >= 0.5. JW Krio, van de Vate, and third-party vocabulary are excluded. Government-source/public-domain status still needs jurisdiction and source-material verification.
- The raw Qwen pretraining corpus is not downloaded or claimed to be available.
- Generated teacher data is not automatically accepted as verified Krio. Native-speaker review and a separate language benchmark are required.

## Baseline (Measured 2026-11-14)

Active multilingual checkpoint: 721,536 parameters, 892-token BPE vocabulary, context 96. Held-out language-ID accuracy: Krio 0.80, English 0.93, Mende 0.97, Temne 0.75. Translation chrF: Krio->English 0.094, English->Krio 0.093, Mende->English 0.095, English->Mende 0.055, Temne->English 0.060, English->Temne 0.105. It abstains on unseen translations by design.

The previous arithmetic distillation experiment did not improve held-out reasoning and was not promoted. No Qwen benchmark results are claimed because the base weights and ML dependencies could not be downloaded in this environment.

## Remaining

1. Complete model and dependency downloads in a network environment that can reach Hugging Face and PyTorch's CPU wheel index.
2. Run the provided Qwen loader smoke test, then prepare the licensed data and inspect its manifest/counts.
3. Perform a short CPU training smoke test only; use a CUDA GPU for meaningful LoRA training.
4. Run held-out evaluation and native-speaker review before making any quality claims or promoting an adapter.
5. Add licensed coding, agricultural, financial-literacy, and locally authored cultural benchmarks before measuring those objectives.

Summit is a Qwen-derived fine-tuned model in this path, not an independently pretrained model. The adapter can run locally without a Qwen service, while its base weights remain Qwen-derived.

## Approved Qwen3-4B Student Path (2026-11-16)

The selected student is `Qwen/Qwen3-4B`, revision
`1cfa9a7208912126459214e8b04321603b3df60c`. Hugging Face model metadata reports 4,022,468,096 parameters and the
repository supplies an Apache-2.0 license. This is approximately 4B rather than exactly 3B; Qwen2.5-3B-Instruct was
not selected because its repository uses the non-commercial Qwen Research License. The model card and license were
inspected before fetching student weights.

The installed `qwen2.5-coder:7b` Ollama model (7.6B, Q4_K_M) is the local teacher. It communicates only with
`127.0.0.1:11434`; it supplies answers, not transferable weights or logits. Summit's student is the compatible Qwen3
checkpoint plus trainable Summit LoRA adapters. The base is frozen during SFT, so Summit remains Qwen-derived and the
adapter is the part whose weights change.

`tools/generate_verified_arithmetic.py` creates bounded arithmetic examples. Each accepted answer is independently
verified with exact integer arithmetic, and the canonical solution trace is constructed locally. The fixed arithmetic
evaluation set is excluded from training. `tools/qwen_training_loop.py` can run one cycle or a capped continuous loop;
it has pause/stop markers, a persistent queue cap, per-cycle logs/status, immutable candidate adapters, and a promotion
gate covering math, conversation, translations, and the large-number regression check.

The student model files are stored outside the repository. The current PC has a CPU-only PyTorch 2.8 environment,
15.7 GB RAM, and no CUDA device. It can generate verified teacher data, but the worker refuses to load/train the 4B
student here. Run SFT on a CUDA system with sufficient free VRAM (the current configuration requires at least 16 GiB).
No continuous Qwen student training is started automatically.