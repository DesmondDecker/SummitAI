"""Optional escalation to a large language model for questions the small local model cannot handle.
OFF by default. Sending user text to an external service is a privacy decision: enable it deliberately,
tell users, and read the provider's data terms."""
import json, os, urllib.error, urllib.request
from urllib.parse import urlsplit


class BackendError(Exception):
    pass


class Backend:
    name = "none"

    def complete(self, messages, system=None, max_tokens=512):
        raise NotImplementedError


class MockBackend(Backend):
    """For tests: echoes a canned answer and records what it was sent."""
    name = "mock"

    def __init__(self, answer="(mock answer)", fail=False):
        self.answer, self.fail, self.calls = answer, fail, []

    def complete(self, messages, system=None, max_tokens=512):
        self.calls.append(dict(messages=messages, system=system))
        if self.fail:
            raise BackendError("mock failure")
        return self.answer


class AnthropicBackend(Backend):
    """Calls the Anthropic Messages API. NOT exercised in this project's tests (no network was available):
    try it on your own machine before relying on it. Needs ANTHROPIC_API_KEY and KMT_BACKEND_MODEL
    (a model id from your Anthropic console)."""
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key=None, model=None, timeout=30):
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("KMT_BACKEND_MODEL")
        self.timeout = timeout
        if not self.key or not self.model:
            raise BackendError("set ANTHROPIC_API_KEY and KMT_BACKEND_MODEL to enable this backend")

    def complete(self, messages, system=None, max_tokens=512):
        body = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            body["system"] = system
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", "x-api-key": self.key,
                                              "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        except Exception as e:                                 # network, HTTP error, bad JSON: never crash the caller
            raise BackendError(f"{type(e).__name__}: {e}") from e


class LocalQwenBackend(Backend):
    """Runs Summit's local Qwen checkpoint and optional local LoRA adapter."""
    name = "qwen-local"
    answer_all = True

    def __init__(self, model_path, adapter_path=None, device="auto"):
        from .qwen_runtime import QwenRuntime
        try:
            self.runtime = QwenRuntime(model_path, adapter_path, device)
        except Exception as exc:
            raise BackendError(f"cannot load local Qwen model: {exc}") from exc

    def complete(self, messages, system=None, max_tokens=512):
        try:
            return self.runtime.respond(messages, system=system, max_new_tokens=max_tokens)
        except Exception as exc:
            raise BackendError(f"local Qwen inference failed: {exc}") from exc


class OllamaQwenBackend(Backend):
    """Uses a Qwen model already installed in the local Ollama service."""
    name = "qwen-ollama"
    answer_all = True
    URL = "http://127.0.0.1:11434/api/chat"

    def __init__(self, model=None, timeout=180, url=None):
        self.model = model or os.environ.get("SUMMIT_QWEN_MODEL", "custom-qwen:latest")
        self.timeout = timeout
        self.url = url or self.URL
        if not self.model.strip():
            raise BackendError("set SUMMIT_QWEN_MODEL to an installed local Ollama Qwen model")
        endpoint = urlsplit(self.url)
        if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
                or endpoint.username or endpoint.password):
            raise BackendError("Qwen Ollama endpoint must use a loopback address")
        if not 1 <= timeout <= 600:
            raise ValueError("Ollama timeout must be between 1 and 600 seconds")

    def complete(self, messages, system=None, max_tokens=512):
        turns = list(messages)
        if system:
            turns.insert(0, {"role": "system", "content": system})
        body = {
            "model": self.model,
            "messages": turns,
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": max_tokens},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackendError(f"local Ollama Qwen request failed: {type(exc).__name__}: {exc}") from exc
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise BackendError("local Ollama Qwen returned an empty or invalid response")
        return content.strip()
