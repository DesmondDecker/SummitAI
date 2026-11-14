"""Per-conversation state: history, language, facts, pending confirmations. In memory only, bounded, expiring."""
import collections, threading, time
from .memory import MemoryStore


class Session:
    def __init__(self, sid, max_history=20):
        self.sid, self.created, self.touched = sid, time.time(), time.time()
        self.history = collections.deque(maxlen=max_history)     # (role, text)
        self.langs = collections.deque(maxlen=5)                  # recent detected languages
        self.facts = None                                         # MemoryStore, created on first fact (saves RAM)
        self.pending = None                                       # e.g. a robot command awaiting confirmation
        self.last_translation = None

    def add(self, role, text):
        self.history.append((role, text)); self.touched = time.time()

    @property
    def lang(self):
        """Most common recent language, or None."""
        return collections.Counter(self.langs).most_common(1)[0][0] if self.langs else None

    def add_fact(self, text):
        if self.facts is None:
            self.facts = MemoryStore(None, dim=512, capacity=16)
        return self.facts.add_fact(text)

    def recall(self, query, k=2):
        return [it["text"] for _, it in self.facts.search(query, k, kind="fact")] if self.facts else []

    def recent(self, n=6):
        return list(self.history)[-n:]


class SessionManager:
    """Thread-safe LRU of sessions with a time-to-live, so memory stays bounded at any user count."""
    def __init__(self, max_sessions=10000, ttl=3600):
        self.max, self.ttl, self._s, self._lock = max_sessions, ttl, collections.OrderedDict(), threading.Lock()

    def get(self, sid):
        now = time.time()
        with self._lock:
            s = self._s.get(sid)
            if s is None or now - s.touched > self.ttl:
                s = Session(sid); self._s[sid] = s
            self._s.move_to_end(sid)
            while len(self._s) > self.max:
                self._s.popitem(last=False)
            return s

    def drop(self, sid):
        with self._lock:
            self._s.pop(sid, None)

    def __len__(self):
        return len(self._s)
