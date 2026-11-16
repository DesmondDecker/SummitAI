"""HTTP API.   python serve.py --ckpt checkpoints --port 8080 [--api-key SECRET] [--robot sim]

POST /v1/chat      {"text", "session", "task"?, "format"?}    context-aware routing (translate / ask / robot / ...)
POST /v1/translate {"text", "src"?, "tgt"?, "format"?}         cached
POST /v1/identify  {"text", "format"?}                         cached
POST /v1/correct   {"src_lang","src","tgt_lang","tgt"}         queued for REVIEW unless --accept-corrections
POST /v1/robot/stop                                            emergency stop (skips rate limits)
GET  /health   GET /metrics
format = json | text | markdown | xml | csv   (query string ?format= or body field)
Standard library only. Stateless apart from bounded in-memory sessions, so you can run many copies behind a load balancer."""
import mimetypes
from pathlib import Path

    def _serve_static(self, rel_path: str, content_type: str = None):
        """Serve a file from the ./static directory.
        rel_path – path relative to the static folder (e.g. "index.html").
        If content_type is None we guess based on the file suffix.
        """
        static_root = Path(__file__).parent / "static"
        file_path = static_root / rel_path
        if not file_path.is_file():
            return self._err(404, f"Static file {rel_path!r} not found", fmt)
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = "application/octet-stream"
        with open(file_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.status = 200
        return

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from kmt.assistant import Assistant
from kmt.backends import AnthropicBackend
from kmt.data import feedback_append, read_tsv
from kmt.dialogue import Dialogue
from kmt.formats import FORMATS, render
from kmt.robot import RobotController, SimRobot
from kmt.session import SessionManager
from voice import stt, tts

# ---------------------------------------------------------------------------
# Generate an introductory audio file (once) using the TTS helper.
# ---------------------------------------------------------------------------
import os
INTRO_TEXT = "Hello, I am Summit AI. I provide multilingual language identification, translation lookup, and robot command simulation."
intro_path = os.path.join(os.path.dirname(__file__), "static", "intro.wav")
if not os.path.isfile(intro_path):
    os.makedirs(os.path.dirname(intro_path), exist_ok=True)
    tts(INTRO_TEXT, out_path=intro_path)


SID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class LRU:
    def __init__(self, n=5000):
        self.n, self.d, self.lock = n, collections.OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k); return self.d[k]

    def put(self, k, v):
        with self.lock:
            self.d[k] = v; self.d.move_to_end(k)
            while len(self.d) > self.n:
                self.d.popitem(last=False)

    def clear(self):
        with self.lock:
            self.d.clear()


class RateLimiter:
    """Token bucket per client key, with a bounded table so it cannot be used to exhaust memory."""
    def __init__(self, rate, burst, max_keys=50000):
        self.rate, self.burst, self.max, self.b, self.lock = rate, burst, max_keys, collections.OrderedDict(), threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self.lock:
            tokens, last = self.b.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self.b[key] = (tokens - 1 if ok else tokens, now); self.b.move_to_end(key)
            while len(self.b) > self.max:
                self.b.popitem(last=False)
            return ok, (0 if ok else (1 - tokens) / self.rate)


class Metrics:
    def __init__(self):
        self.lock, self.c, self.lat, self.t0 = threading.Lock(), collections.Counter(), collections.deque(maxlen=5000), time.time()

    def record(self, path, status, ms):
        with self.lock:
            self.c["requests"] += 1; self.c[f"status_{status}"] += 1; self.c[f"path_{path}"] += 1; self.lat.append(ms)

    def snapshot(self):
        with self.lock:
            lat = sorted(self.lat); q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 2) if lat else None
            return dict(uptime_s=round(time.time() - self.t0), counters=dict(self.c), latency_ms=dict(p50=q(.5), p95=q(.95), p99=q(.99), samples=len(lat)))


def load_lexicon(data_dir):
    """data/robot_lexicon.tsv: local_word<TAB>english_word. Extends what the robot parser understands (e.g. extra
    STOP words). Strictly validated: both sides must be single alphabetic words, otherwise the server refuses to start."""
    lex = {}
    for w, c in read_tsv(os.path.join(data_dir, "robot_lexicon.tsv"), 2):
        if not (re.fullmatch(r"[^\W\d_]+", w) and re.fullmatch(r"[a-z]+", c.lower())):
            raise SystemExit(f"robot_lexicon.tsv: invalid entry {w!r} -> {c!r} (single alphabetic words only)")
        lex[w.lower()] = c.lower()
    return lex


class App:
    def __init__(self, a):
        self.bot = Assistant(a.ckpt, a.memory, a.data)
        robot = RobotController(SimRobot(), lexicon=load_lexicon(a.data)) if a.robot == "sim" else None
        backend = None
        if a.backend == "anthropic":
            backend = AnthropicBackend()
        self.dialogue = Dialogue(self.bot, robot, backend, SessionManager(a.max_sessions, a.session_ttl), a.max_chars)
        self.robot, self.api_key, self.max_chars, self.max_body = robot, a.api_key, a.max_chars, a.max_body
        self.accept_corrections, self.pending_path = a.accept_corrections, os.path.join(a.memory, "pending_corrections.jsonl")
        self.cache, self.limiter, self.metrics = LRU(a.cache_size), RateLimiter(a.rate, a.burst), Metrics()
        self.inflight, self.write_lock = threading.BoundedSemaphore(a.max_inflight), threading.Lock()
        self.retry_after = 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                      # keep-alive: no reconnect cost per request
    timeout = 10                                       # drop slow/stalled clients
    server_version = "KMT"

    def log_message(self, *a):                         # never log request bodies or user text
        pass

    # ---- helpers ------------------------------------------------------
    def _send(self, code, obj, fmt="json", headers=None):
        try:
            ct, body = render(obj, fmt)
        except ValueError:
            ct, body = render(obj, "json")
        data = body.encode("utf-8")
        self.send_response(code)
        for k, v in {"Content-Type": ct, "Content-Length": str(len(data)), "Cache-Control": "no-store",
                     "X-Content-Type-Options": "nosniff", **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(data)
        self.status = code

    def _err(self, code, msg, fmt="json", headers=None):
        self._send(code, dict(error=msg), fmt, headers)

    def _authorized(self):
        key = self.server.app.api_key
        if not key:
            return True
        got = self.headers.get("Authorization", "")
        return got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode(), key.encode())

    def _body(self, fmt):
        n = self.headers.get("Content-Length")
        if n is None or not n.isdigit():
            self._err(411, "Content-Length required", fmt); return None
        if int(n) > self.server.app.max_body:
            self._err(413, "request too large", fmt); return None
        try:
            obj = json.loads(self.rfile.read(int(n)).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._err(400, "body must be valid UTF-8 JSON", fmt); return None
        if not isinstance(obj, dict):
            self._err(400, "body must be a JSON object", fmt); return None
        return obj

    def _text(self, body, key, fmt, required=True):
        v = body.get(key)
        if v is None and not required:
            return ""
        if not isinstance(v, str) or (required and not v.strip()):
            self._err(400, f"'{key}' must be a non-empty string", fmt); return None
        if len(v) > self.server.app.max_chars:
            self._err(400, f"'{key}' is longer than {self.server.app.max_chars} characters", fmt); return None
        return v

    # ---- dispatch -----------------------------------------------------
        def do_GET(self):
        # Serve static files (root page and intro audio) before other routes
        if self.path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if self.path == "/intro.wav":
            return self._serve_static("intro.wav")
        self._handle("GET")
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method):
        t0, app = time.perf_counter(), self.server.app
        u = urlparse(self.path); path = u.path.rstrip("/") or "/"
        fmt = (parse_qs(u.query).get("format", ["json"])[0]).lower()
        self.status = 500
        try:
            if fmt not in FORMATS:
                return self._err(400, f"format must be one of {', '.join(FORMATS)}")
            if path == "/health" and method == "GET":
                return self._send(200, dict(status="ok", langs=app.bot.langs, robot=bool(app.robot)), fmt)
            if not self._authorized():
                return self._err(401, "missing or invalid API key", fmt, {"WWW-Authenticate": "Bearer"})
            ok, wait = app.limiter.allow(self.headers.get("Authorization") or self.client_address[0])
            if not ok and path != "/v1/robot/stop":
                return self._err(429, "rate limit exceeded", fmt, {"Retry-After": str(max(1, int(wait + 0.99)))})
            if path == "/metrics" and method == "GET":
                return self._send(200, app.metrics.snapshot(), fmt)
            if path == "/v1/robot/stop" and method == "POST":
                if not app.robot:
                    return self._err(404, "robot control is not enabled", fmt)
                return self._send(200, app.robot.emergency_stop("http"), fmt)
            if method != "POST" or path not in ("/v1/chat", "/v1/translate", "/v1/identify", "/v1/correct"):
                return self._err(404, "not found", fmt)
            body = self._body(fmt)
            if body is None:
                return
            fmt = (body.get("format") or fmt).lower()
            if fmt not in FORMATS:
                return self._err(400, f"format must be one of {', '.join(FORMATS)}")
            if not app.inflight.acquire(blocking=False):                 # overloaded: shed load instead of queueing forever
                return self._err(503, "server busy, retry shortly", fmt, {"Retry-After": "1"})
            try:
                self._route(path, body, fmt)
            finally:
                app.inflight.release()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc(file=sys.stderr)                         # details to the server log only, never to the client
            try:
                self._err(500, "internal error", fmt)
            except Exception:
                pass
        finally:
            app.metrics.record(path if path.startswith(("/v1", "/health", "/metrics")) else "other", self.status, (time.perf_counter() - t0) * 1000)

    def _route(self, path, body, fmt):
        app = self.server.app
        # ---- new audio endpoint -------------------------------------------------
        if path == "/v1/audio":
            # Expect raw 16kHz mono PCM WAV data in the request body.
            # The body has already been read into *body* (as dict) for JSON routes,
            # but for binary we need to read directly from the underlying stream.
            # We'll re‑read the raw payload using the Content‑Length header.
            # This block bypasses the usual JSON parsing because the client
            # sends a binary wav file.
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n == 0:
                    return self._err(411, "Content-Length required", fmt)
                raw = self.rfile.read(n)
                # Write to a temporary file so the STT/TTS helpers can work with a path.
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                # Speech‑to‑text (offline or online depending on VOICE_MODE)
                text = stt(tmp_path)
                os.remove(tmp_path)
                # Re‑use the existing chat logic – we call the same function that
                # the /v1/chat endpoint uses.  The Assistant is accessible via
                # app.dialogue.bot (or app.bot in older code).  Here we call the
                # dialogue directly for consistency.
                # Create a temporary session ID for this audio call.
                sid = f"audio-{self.client_address[0]}-{int(time.time())}"
                # Use the same default task handling as chat – "auto".
                r = app.dialogue.respond(text, sid, "auto", None, None, None)
                reply_text = r.get("text") or r.get("response") or ""
                # Text‑to‑speech (offline/online)
                wav_out = tts(reply_text)
                # Return the generated wav as binary response (Content‑Type: audio/wav)
                with open(wav_out, "rb") as f:
                    wav_bytes = f.read()
                # Clean up the generated file
                os.remove(wav_out)
                # Send binary response – we bypass render() because it expects JSON.
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
                self.status = 200
                return
            except Exception as exc:
                # Log server‑side error but return generic message to client.
                traceback.print_exc(file=sys.stderr)
                return self._err(500, f"audio processing error: {exc}", fmt)
        # ---- existing routes ---------------------------------------------------
        app = self.server.app
        if path == "/v1/correct":
            f = {k: self._text(body, k, fmt) for k in ("src_lang", "src", "tgt_lang", "tgt")}
            if any(v is None for v in f.values()):
                return
            rec = dict(task="translate", **f)
            if app.accept_corrections:
                with app.write_lock:
                    app.bot.correct_translation(f["src_lang"], f["src"], f["tgt_lang"], f["tgt"])
                app.cache.clear()
                return self._send(200, dict(status="applied"), fmt)
            feedback_append(app.pending_path, rec)
            return self._send(202, dict(status="queued_for_review"), fmt)
        text = self._text(body, "text", fmt)
        if text is None:
            return
        if path == "/v1/identify":
            key = ("id", text); r = app.cache.get(key)
            if r is None:
                best, probs = app.bot.identify(text)
                r = dict(lang=best, probabilities={k: round(v, 4) for k, v in probs.items()}); app.cache.put(key, r)
            return self._send(200, r, fmt)
        if path == "/v1/translate":
            src, tgt = body.get("src"), body.get("tgt", "en")
            key = ("tr", text, src, tgt); r = app.cache.get(key); hit = r is not None
            if r is None:
                res = app.bot.translate(text, src, tgt)
                r = dict(text=res["text"], **{k: res[k] for k in ("src_lang", "tgt_lang", "source", "confidence", "warning", "chunks", "alternatives", "suggestions", "matched", "match") if k in res})
                app.cache.put(key, r)
            return self._send(200, dict(r, cached=hit), fmt)
        sid = body.get("session", f"anon-{self.client_address[0]}")
        if not isinstance(sid, str) or not SID.match(sid):
            return self._err(400, "'session' must be 1-64 characters of letters, digits, _ . : -", fmt)
        task = body.get("task", "auto")
        if task not in ("auto", "translate", "ask", "identify", "robot", "remember"):
            return self._err(400, "unknown task", fmt)
        r = app.dialogue.respond(text, sid, task, body.get("lang"), body.get("src"), body.get("tgt"))
        self._send(200, r, fmt)
        app = self.server.app
        if path == "/v1/correct":
            f = {k: self._text(body, k, fmt) for k in ("src_lang", "src", "tgt_lang", "tgt")}
            if any(v is None for v in f.values()):
                return
            rec = dict(task="translate", **f)
            if app.accept_corrections:
                with app.write_lock:
                    app.bot.correct_translation(f["src_lang"], f["src"], f["tgt_lang"], f["tgt"])
                app.cache.clear()
                return self._send(200, dict(status="applied"), fmt)
            feedback_append(app.pending_path, rec)
            return self._send(202, dict(status="queued_for_review"), fmt)
        text = self._text(body, "text", fmt)
        if text is None:
            return
        if path == "/v1/identify":
            key = ("id", text); r = app.cache.get(key)
            if r is None:
                best, probs = app.bot.identify(text)
                r = dict(lang=best, probabilities={k: round(v, 4) for k, v in probs.items()}); app.cache.put(key, r)
            return self._send(200, r, fmt)
        if path == "/v1/translate":
            src, tgt = body.get("src"), body.get("tgt", "en")
            key = ("tr", text, src, tgt); r = app.cache.get(key); hit = r is not None
            if r is None:
                res = app.bot.translate(text, src, tgt)
                r = dict(text=res["text"], **{k: res[k] for k in ("src_lang", "tgt_lang", "source", "confidence", "warning", "chunks", "alternatives", "suggestions", "matched", "match") if k in res})
                app.cache.put(key, r)
            return self._send(200, dict(r, cached=hit), fmt)
        sid = body.get("session", f"anon-{self.client_address[0]}")
        if not isinstance(sid, str) or not SID.match(sid):
            return self._err(400, "'session' must be 1-64 characters of letters, digits, _ . : -", fmt)
        task = body.get("task", "auto")
        if task not in ("auto", "translate", "ask", "identify", "robot", "remember"):
            return self._err(400, "unknown task", fmt)
        r = app.dialogue.respond(text, sid, task, body.get("lang"), body.get("src"), body.get("tgt"))
        self._send(200, r, fmt)


def build_server(a):
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    srv.app = App(a)
    return srv


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints"); ap.add_argument("--memory", default="memory"); ap.add_argument("--data", default="data")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("KMT_API_KEY"), help="require 'Authorization: Bearer <key>' (or set KMT_API_KEY)")
    ap.add_argument("--rate", type=float, default=20, help="requests/second per client"); ap.add_argument("--burst", type=float, default=40)
    ap.add_argument("--max-inflight", type=int, default=8, help="concurrent model computations before answering 503")
    ap.add_argument("--cache-size", type=int, default=5000); ap.add_argument("--max-sessions", type=int, default=10000)
    ap.add_argument("--session-ttl", type=int, default=3600); ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--max-body", type=int, default=16384)
    ap.add_argument("--accept-corrections", action="store_true", help="apply corrections to memory immediately (trusted/private use only)")
    ap.add_argument("--robot", choices=["none", "sim"], default="none"); ap.add_argument("--backend", choices=["none", "anthropic"], default="none")
    ap.add_argument("--allow-unauthenticated-robot", action="store_true")
    a = ap.parse_args(argv)
    if a.robot != "none" and not a.api_key and not a.allow_unauthenticated_robot:
        ap.error("robot control requires --api-key (anyone who can reach the server could move the robot); "
                 "use --allow-unauthenticated-robot only on a private test setup")
    return a


if __name__ == "__main__":
    a = parse_args(); srv = build_server(a)
    print(f"KMT serving on http://{a.host}:{a.port}  (auth {'on' if a.api_key else 'OFF'}, robot {a.robot}, backend {a.backend})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down"); srv.shutdown()
