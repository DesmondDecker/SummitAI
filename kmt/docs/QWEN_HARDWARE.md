# Qwen Hardware and Setup

## Observed Workstation

- Windows, Python 3.13.14, NumPy 2.5.4
- Intel Core i7-1355U, 10 cores / 12 logical processors
- 15.7 GB physical RAM
- Intel Iris Xe integrated graphics; no CUDA GPU detected
- Disk space varied during the session; verify at least 8 GB free before downloading weights and installing the isolated environment

## Expected Use

- Qwen3-0.6B Base weight files are about 1.2 GB in BF16 on Hugging Face. CPU inference in float32 needs additional RAM for weights, activations, and Python libraries; keep a modest context and batch size.
- CPU LoRA training is supported for smoke tests, not recommended for meaningful training speed. Use a CUDA GPU with sufficient VRAM for longer runs.
- The current setup pins CPU-only PyTorch 2.8.0 and uses a separate `.venv-qwen`, leaving the NumPy environment unchanged.
- Network access to Hugging Face and `download.pytorch.org` is required for initial setup. The current attempt timed out before downloading weights or installing packages.

## Qwen3-4B Summit Student

The approved Qwen3-4B checkpoint contains 4,022,468,096 parameters and is licensed Apache-2.0. BF16 weights alone
require about 8 GB; fine-tuning additionally needs activation and adapter/optimizer memory. `tools/qwen_training_loop.py`
requires a CUDA device with at least 16 GiB currently free before it begins any training cycle. A 24 GiB GPU is preferred
for longer contexts or larger batches. The workstation above has no CUDA GPU, so use it for local Qwen teacher generation
and data preparation only; do not run 4B SFT on its CPU.

The installed Ollama teacher is `qwen2.5-coder:7b` (7.6B Q4_K_M), queried on loopback. The Qwen3-4B student is a separate
Transformers checkpoint plus Summit LoRA adapter. Ollama quantized teacher files are not used as student weights.

## Safe Setup Commands

```powershell
python -m venv .venv-qwen
.\.venv-qwen\Scripts\python.exe -m pip install -r requirements-qwen-cpu.txt
hf download Qwen/Qwen3-0.6B-Base --max-workers 1 --local-dir "$env:LOCALAPPDATA\SummitAI\models\Qwen3-0.6B-Base"
```

After the download, verify required files (`config.json`, `model.safetensors`, tokenizer files) exist before running `chat_qwen.py`. Do not point the Qwen loader at the NumPy `.npz` checkpoints.