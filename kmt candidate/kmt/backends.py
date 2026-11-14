"""Optional escalation to a large language model for questions the small local model cannot handle.
OFF by default. Sending user text to an external service is a privacy decision: enable it deliberately,
tell users, and read the provider's data terms."""
import json, os, urllib.request


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
