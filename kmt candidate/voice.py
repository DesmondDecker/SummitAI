# -------------------------------------------------
# voice.py – unified Speech‑to‑Text / Text‑to‑Speech helper
# -------------------------------------------------
"""Provides a simple API that works **both offline** (Vosk + Coqui TTS)
and **online** (OpenAI Whisper & Azure Speech) depending on configuration.

Typical usage::

    from voice import stt, tts
    text = stt("audio.wav")
    wav_path = tts(text)

Both functions return ``str`` – the transcribed text and the path to a generated
WAV file respectively.

Configuration is driven by the environment variable ``VOICE_MODE``:

* ``offline`` – force the local models (default when the variable is unset).
* ``online``  – attempt to use the online services.  If an online service
  fails (e.g. no API key or no network), the code falls back to the offline
  implementation automatically.

You must place the required models under ``<project>/models/``:

* Vosk English model (e.g. ``vosk-en-0.22``) – download from
  https://alphacephei.com/vosk/models and unzip.
* Coqui TTS model – the default ``tts_models/en/ljspeech/tacotron2-DDC``
  is downloaded automatically on first use.

Online services require API keys set in environment variables:

* OpenAI Whisper – ``OPENAI_API_KEY``
* Azure Speech – ``AZURE_SPEECH_KEY`` and ``AZURE_SPEECH_REGION``
"""

import os
import json
import pathlib
import subprocess
from typing import Callable

# ---------------------------------------------------------------------------
# Helper to decide which mode to run in
# ---------------------------------------------------------------------------
def _is_online_requested() -> bool:
    """Return ``True`` when the user explicitly requests online mode.
    The environment variable ``VOICE_MODE`` can be set to ``online`` or ``offline``.
    Any other value (including missing) defaults to ``offline`` for safety.
    """
    return os.getenv("VOICE_MODE", "offline").lower() == "online"

# ---------------------------------------------------------------------------
# Offline implementations (Vosk for STT, Coqui TTS for TTS)
# ---------------------------------------------------------------------------
# Vosk imports are optional – they may not be installed when the user only wants
# the online mode.  Import lazily inside the functions.

def _offline_stt(audio_path: str) -> str:
    """Transcribe *audio_path* (wav, 16kHz mono) using Vosk.
    Returns the plain text transcript.
    """
    try:
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        raise RuntimeError("Vosk library not available – install with 'pip install vosk'" ) from e

    # Resolve model directory – we expect it under <project>/models/vosk‑en‑0.22
    project_dir = pathlib.Path(__file__).parent.parent  # <project>/kmt candidate -> <project>
    model_dir = project_dir / "models" / "vosk-en-0.22"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Vosk model not found at {model_dir}. Download from https://alphacephei.com/vosk/models"
        )
    model = Model(str(model_dir))

    import wave
import numpy as np

# Helper to read a wav file into a NumPy int16 array
def _wav_to_numpy(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
        data = np.frombuffer(audio_bytes, dtype='<i2')  # little‑endian int16
        # If the file has multiple channels, keep only the first one
        if wf.getnchannels() > 1:
            data = data[:: wf.getnchannels()]
        return data, sr

    wf = wave.open(audio_path, "rb")
    if wf.getframerate() != 16000 or wf.getnchannels() != 1:
        raise ValueError("Vosk requires 16kHz mono PCM WAV files")

    recognizer = KaldiRecognizer(model, wf.getframerate())
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        recognizer.AcceptWaveform(data)
    result = recognizer.FinalResult()
    wf.close()
    return json.loads(result).get("text", "")


def _offline_tts(text: str, out_path: str = "output.wav") -> str:
    """Generate a wav file from *text* using a lightweight TTS approach.

    Tries to use pyttsx3 (offline, no network). If unavailable,
    falls back to Windows PowerShell SAPI5 synthesis.
    """
    # Try pyttsx3 first
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return os.path.abspath(out_path)
    except Exception:
        # Fallback: use PowerShell SAPI5
        ps_script = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{out_path}'); "
            f"$s.Speak('{text}');"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate TTS via PowerShell: {result.stderr}")
        return os.path.abspath(out_path)

# ---------------------------------------------------------------------------
# Online implementations (OpenAI Whisper for STT, Azure Speech for TTS)
# ---------------------------------------------------------------------------
def _online_stt(audio_path: str) -> str:
    """Transcribe using OpenAI Whisper (requires ``OPENAI_API_KEY``).
    Falls back to offline if the key is missing or the request fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for online STT")
    try:
        import openai
        openai.api_key = api_key
        # Whisper expects a file‑like object; we open in binary mode.
        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
        # The response object has .text attribute
        return transcript.text.strip()
    except Exception as exc:
        raise RuntimeError(f"OpenAI Whisper failed: {exc}") from exc


def _online_tts(text: str, out_path: str = "output.wav") -> str:
    """Synthesize speech via Azure Cognitive Services (requires key & region).
    Falls back to offline if the service is unavailable.
    """
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not (key and region):
        raise RuntimeError("Azure Speech credentials not set (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)")
    try:
        import azure.cognitiveservices.speech as speechsdk
    except Exception as e:
        raise RuntimeError("azure-cognitiveservices-speech package missing – install with 'pip install azure-cognitiveservices-speech'") from e

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS failed: {result.reason}")
    return os.path.abspath(out_path)

# ---------------------------------------------------------------------------
# Public wrapper functions – they decide at runtime which implementation to use.
# ---------------------------------------------------------------------------
def stt(audio_path: str) -> str:
    """Speech‑to‑Text entry point.
    Chooses offline or online based on ``VOICE_MODE`` and availability of the
    required credentials.
    """
    if _is_online_requested():
        try:
            return _online_stt(audio_path)
        except Exception as e:
            print(f"Online STT failed ({e}); falling back to offline.")
    # offline path (always available if the model files exist)
    return _offline_stt(audio_path)


def tts(text: str, out_path: str = "output.wav") -> str:
    """Text‑to‑Speech entry point.
    Same fallback logic as :func:`stt`.
    """
    if _is_online_requested():
        try:
            return _online_tts(text, out_path)
        except Exception as e:
            print(f"Online TTS failed ({e}); falling back to offline.")
    return _offline_tts(text, out_path)

# ---------------------------------------------------------------------------
# Convenience: a simple voice‑chat loop for the CLI (uses the same wrappers).
# ---------------------------------------------------------------------------
import queue
import threading
import numpy as np
import sounddevice as sd

def voice_chat_loop(prompt_callback: Callable[[str], str]):
    """Continuously record from the default microphone, send the recognised text
    to *prompt_callback*, synthesize the reply and play it back.

    *prompt_callback* should accept a ``str`` (the user utterance) and return a
    ``str`` (the system response).  For example you can pass the ``chat`` function
    from ``chat.py``.
    """
    sample_rate = 16000
    blocksize = 8000  # ~0.5 s chunks
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _audio_cb(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        blocksize=blocksize,
                        callback=_audio_cb):
        print("Listening – press Ctrl‑C to stop")
        audio_chunks = []
        try:
            while True:
                chunk = q.get()
                audio_chunks.append(chunk.squeeze())
                # When we have roughly 1 second of audio, process it.
                if len(audio_chunks) * blocksize >= sample_rate:
                    audio_np = np.concatenate(audio_chunks)
                    # Save temporary wav for the STT functions (both offline & online accept a path)
                    tmp_path = "temp_input.wav"
                    sd.wait()  # ensure no dangling audio buffers
                    sd.rec(0)  # dummy call to avoid PyAudio warnings
                    import wave
import numpy as np

# Helper to read a wav file into a NumPy int16 array
def _wav_to_numpy(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
        data = np.frombuffer(audio_bytes, dtype='<i2')  # little‑endian int16
        # If the file has multiple channels, keep only the first one
        if wf.getnchannels() > 1:
            data = data[:: wf.getnchannels()]
        return data, sr

                    with wave.open(tmp_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # int16
    wf.setframerate(sample_rate)
    wf.writeframes(audio_np.tobytes())
                    user_text = stt(tmp_path)
                    print(f"\nYou said: {user_text}")
                    reply = prompt_callback(user_text)
                    print(f"AI: {reply}")
                    wav_out = tts(reply)
                    # Playback the generated wav
                    import wave
import numpy as np

with wave.open(wav_out, 'rb') as wf:
    sr = wf.getframerate()
    n_frames = wf.getnframes()
    audio_bytes = wf.readframes(n_frames)
    # Convert little‑endian 16‑bit PCM to numpy array
    data = np.frombuffer(audio_bytes, dtype='<i2')
    # If stereo, take first channel
    if wf.getnchannels() > 1:
        data = data[::wf.getnchannels()]
                    sd.play(data, sr)
                    sd.wait()
                    audio_chunks.clear()
        except KeyboardInterrupt:
            print("\nVoice chat stopped.")

# End of file
