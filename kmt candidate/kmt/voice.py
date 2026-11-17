"""Local English speech recognition through the Windows speech engine."""
import base64
import io
import json
import shutil
import subprocess
import sys
import wave


class SpeechRecognitionError(Exception):
    pass


_RECOGNIZE_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$audio = New-Object System.IO.MemoryStream
[Console]::OpenStandardInput().CopyTo($audio)
$audio.Position = 0
$engine = $null
try {
    $culture = [Globalization.CultureInfo]::GetCultureInfo('en-US')
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture)
    $engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $engine.InitialSilenceTimeout = [TimeSpan]::FromSeconds(4)
    $engine.BabbleTimeout = [TimeSpan]::FromSeconds(4)
    $engine.EndSilenceTimeout = [TimeSpan]::FromSeconds(0.7)
    $engine.SetInputToWaveStream($audio)
    $result = $engine.Recognize([TimeSpan]::FromSeconds(20))
    if ($null -eq $result -or [string]::IsNullOrWhiteSpace($result.Text)) {
        [Console]::Error.WriteLine('No clear speech was detected. Try again in a quieter place.')
        exit 2
    }
    [Console]::Out.WriteLine((ConvertTo-Json -InputObject $result.Text -Compress))
}
finally {
    if ($null -ne $engine) { $engine.Dispose() }
    $audio.Dispose()
}
"""


class WindowsSpeechRecognizer:
    sample_rate = 16000
    max_duration_seconds = 15
    max_audio_bytes = 512 * 1024

    def __init__(self):
        self.executable = shutil.which("powershell.exe") or shutil.which("powershell")
        self.available = sys.platform == "win32" and self.executable is not None

    def transcribe(self, audio):
        if not self.available:
            raise SpeechRecognitionError("Local English speech recognition is unavailable on this server.")
        if not isinstance(audio, bytes) or len(audio) > self.max_audio_bytes:
            raise SpeechRecognitionError("The voice recording is too large.")
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if (wav.getnchannels() != 1 or wav.getsampwidth() != 2 or
                        wav.getframerate() != self.sample_rate or wav.getcomptype() != "NONE"):
                    raise SpeechRecognitionError("The recording must be uncompressed, 16 kHz mono audio.")
                duration = wav.getnframes() / wav.getframerate()
                if duration < 0.25 or duration > self.max_duration_seconds:
                    raise SpeechRecognitionError("Record between a quarter second and 15 seconds of speech.")
        except (wave.Error, EOFError) as exc:
            raise SpeechRecognitionError("The voice recording is not a valid WAV file.") from exc

        encoded_script = base64.b64encode(_RECOGNIZE_SCRIPT.encode("utf-16le")).decode("ascii")
        try:
            result = subprocess.run(
                [self.executable, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_script],
                input=audio,
                capture_output=True,
                timeout=25,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise SpeechRecognitionError("Local speech recognition timed out. Please try a shorter recording.") from exc
        except OSError as exc:
            raise SpeechRecognitionError("Could not start the local speech recognizer.") from exc

        if result.returncode:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            if error:
                raise SpeechRecognitionError(error[-500:])
            raise SpeechRecognitionError("Local speech recognition could not process that recording.")
        try:
            transcript = json.loads(result.stdout.decode("utf-8").strip())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpeechRecognitionError("The local speech recognizer returned an invalid response.") from exc
        if not isinstance(transcript, str) or not transcript.strip():
            raise SpeechRecognitionError("No clear speech was detected. Try again in a quieter place.")
        return transcript.strip()
