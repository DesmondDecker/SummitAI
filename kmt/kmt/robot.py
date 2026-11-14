"""Robot command layer: natural language in, SAFE deterministic commands out.

PRINCIPLE: the language model never produces motor commands. It may only TRANSLATE text. A small, deterministic,
testable parser turns English text into one of a few allow-listed commands; a validator enforces hard limits; risky or
translated commands need explicit confirmation; stop always wins. A real robot must ALSO have hardware safety
(physical e-stop, speed/torque limits in firmware, bumpers). Nothing here replaces that.
"""
import collections, re, threading, time
from dataclasses import dataclass, field

LIMITS = dict(min_move_m=0.05, max_move_m=5.0, min_turn_deg=1.0, max_turn_deg=360.0,
              min_speed=0.05, max_speed=0.5, confirm_move_m=2.0, max_cmds_per_10s=10, confirm_ttl_s=30)
STOP_WORDS = {"stop", "halt", "freeze", "abort", "emergency", "estop", "e-stop"}
YES = {"yes", "yeah", "yep", "confirm", "confirmed", "ok", "okay", "proceed", "sure"}
NO = {"no", "cancel", "nevermind", "negative"}
_UNITS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                  "sixteen seventeen eighteen nineteen".split(), range(20)))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
_NUM = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")                      # ASCII digits only: no exponents, inf or nan
_LEN = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "cm": 0.01, "centimeter": 0.01,
        "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01}
_ANG = {"degree", "degrees", "deg", "°"}


@dataclass(frozen=True)
class Command:
    action: str                                   # move | turn | stop | set_speed | status | pick_up | put_down
    params: dict = field(default_factory=dict)
    assumed: bool = False                         # True if the parser had to assume something (e.g. a unit)

    def describe(self):
        p = self.params
        return {"move": f"move {'forward' if p.get('meters', 0) >= 0 else 'backward'} {abs(p.get('meters', 0)):g} m",
                "turn": f"turn {'left' if p.get('degrees', 0) > 0 else 'right'} {abs(p.get('degrees', 0)):g} degrees",
                "set_speed": f"set speed to {p.get('speed', 0):g} m/s", "stop": "STOP", "status": "report status",
                "pick_up": "pick up the object in front", "put_down": "put down the held object"}[self.action]


def tokenize(text):
    return re.findall(r"[0-9]+(?:\.[0-9]+)?|[^\W\d_]+(?:-[^\W\d_]+)?|°", (text or "").lower())


def apply_lexicon(tokens, lexicon):
    return [lexicon.get(t, t) for t in tokens] if lexicon else tokens


def _quantity(tokens, start=0):
    """First number at/after `start` -> (value, index_after, unit_token_or_None). Words: 'twenty five', 'half'."""
    i = start
    while i < len(tokens):
        t = tokens[i]
        if _NUM.match(t):
            v = float(t); j = i + 1
        elif t == "half":
            v, j = 0.5, i + 1
        elif t in _UNITS or t in _TENS or t == "hundred":
            v = 0; cur = 0; j = i
            while j < len(tokens) and (tokens[j] in _UNITS or tokens[j] in _TENS or tokens[j] == "hundred"):
                w = tokens[j]
                cur = cur * 100 if (w == "hundred" and cur) else (100 if w == "hundred" else cur + (_UNITS.get(w) if w in _UNITS else _TENS[w]))
                j += 1
            v = float(cur)
        else:
            i += 1; continue
        unit = tokens[j] if j < len(tokens) else None
        return v, j, unit
    return None


@dataclass
class Parsed:
    command: Command = None
    clarify: str = None                            # a question to ask instead of guessing


_NEG = re.compile(r"\b(not|never|no|dont|don'?t|do n'?t|cant|can'?t|cannot|without|instead|except|unless)\b")


def parse(text, lexicon=None):
    """English text -> Parsed(command) | Parsed(clarify=question) | Parsed() if it is not a robot command."""
    toks = apply_lexicon(tokenize(text), lexicon)
    if not toks:
        return Parsed()
    if any(t in STOP_WORDS for t in toks):                        # a false stop is safe, a missed stop is not
        return Parsed(Command("stop"))
    has = lambda *w: any(t in w for t in toks)
    kinds = []
    if has("turn", "rotate", "spin", "u-turn"): kinds.append("turn")
    if has("go", "move", "walk", "drive", "advance", "forward", "backward", "backwards", "reverse", "ahead"):
        kinds.append("move")
    if has("speed"): kinds.append("set_speed")
    if has("status", "battery", "position") or " ".join(toks).startswith("where are you"): kinds.append("status")
    if "pick" in toks or has("grab", "lift"): kinds.append("pick_up")
    if has("drop", "release") or ("put" in toks and "down" in toks): kinds.append("put_down")
    if not kinds:
        return Parsed()
    if _NEG.search((text or "").lower()):                        # "do NOT go forward" must never run as "go forward"
        return Parsed(clarify="I can't tell what you want from a negative sentence. Please state the command directly.")
    if len(kinds) > 1:
        return Parsed(clarify="Please give one command at a time.")
    k = kinds[0]
    if k in ("status", "pick_up", "put_down"):
        return Parsed(Command(k))
    q = _quantity(toks)
    if k == "move":
        back = has("back", "backward", "backwards", "reverse")
        fwd = has("forward", "ahead", "straight", "advance")
        if back == fwd:
            return Parsed(clarify="Forward or backward?")
        if q is None:
            return Parsed(clarify="How far?")
        v, _, unit = q
        if unit in _LEN:
            meters, assumed = v * _LEN[unit], False
        else:
            meters, assumed = v, True                             # no unit: assume metres, but force confirmation
        return Parsed(Command("move", {"meters": meters if fwd else -meters}, assumed))
    if k == "turn":
        if has("around", "u-turn"):
            return Parsed(Command("turn", {"degrees": -180.0}))
        left, right = has("left", "anticlockwise", "counterclockwise"), has("right", "clockwise")
        if left == right:
            return Parsed(clarify="Left or right?")
        deg, assumed = (90.0, False) if q is None else (q[0], False)
        if q is not None and q[2] not in _ANG and q[2] is not None and q[2] in _LEN:
            return Parsed(clarify="How many degrees?")
        return Parsed(Command("turn", {"degrees": deg if left else -deg}, assumed))
    if q is None:                                                 # set_speed
        return Parsed(clarify="What speed in metres per second?")
    return Parsed(Command("set_speed", {"speed": q[0]}))


def validate(cmd, limits=LIMITS):
    """Hard limits. Returns (ok, reason)."""
    p = cmd.params
    if cmd.action == "move":
        m = abs(p.get("meters", float("nan")))
        if not (limits["min_move_m"] <= m <= limits["max_move_m"]):
            return False, f"distance must be between {limits['min_move_m']} and {limits['max_move_m']} metres"
    elif cmd.action == "turn":
        d = abs(p.get("degrees", float("nan")))
        if not (limits["min_turn_deg"] <= d <= limits["max_turn_deg"]):
            return False, f"angle must be between {limits['min_turn_deg']:g} and {limits['max_turn_deg']:g} degrees"
    elif cmd.action == "set_speed":
        v = p.get("speed", float("nan"))
        if not (limits["min_speed"] <= v <= limits["max_speed"]):
            return False, f"speed must be between {limits['min_speed']} and {limits['max_speed']} m/s"
    elif cmd.action not in ("stop", "status", "pick_up", "put_down"):
        return False, "unknown action"
    return True, ""


def needs_confirmation(cmd, from_translation, limits=LIMITS):
    if cmd.action in ("stop", "status"):
        return False
    return (from_translation or cmd.assumed or cmd.action in ("pick_up", "put_down")
            or (cmd.action == "move" and abs(cmd.params["meters"]) >= limits["confirm_move_m"]))


def is_confirmation(text, lexicon=None):
    """A short yes/no answer (English, or local words mapped through the lexicon)."""
    toks = apply_lexicon(tokenize(text), lexicon)
    return 0 < len(toks) <= 3 and toks[0] in YES | NO


# ------------------------------------------------------------------ adapters
class RobotAdapter:
    """Implement this for real hardware. The adapter must enforce its own hard limits independently."""
    def execute(self, cmd):
        raise NotImplementedError

    def emergency_stop(self):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


class SimRobot(RobotAdapter):
    """A 2-D simulated robot with a rectangular safe area (geofence)."""
    def __init__(self, half_size=10.0, limits=LIMITS):
        self.x = self.y = self.heading = 0.0; self.speed = 0.2; self.holding = False
        self.half, self.limits, self.log, self.stopped = half_size, limits, [], False

    def execute(self, cmd):
        ok, why = validate(cmd, self.limits)                      # independent second check at the hardware boundary
        if not ok:
            return dict(ok=False, reason=why)
        import math
        if cmd.action == "move":
            d = cmd.params["meters"]; h = math.radians(self.heading)
            nx, ny = self.x + d * math.cos(h), self.y + d * math.sin(h)
            if abs(nx) > self.half or abs(ny) > self.half:
                return dict(ok=False, reason="that would leave the allowed area")
            self.x, self.y = nx, ny
        elif cmd.action == "turn":
            self.heading = (self.heading + cmd.params["degrees"]) % 360
        elif cmd.action == "set_speed":
            self.speed = cmd.params["speed"]
        elif cmd.action == "pick_up":
            if self.holding: return dict(ok=False, reason="already holding something")
            self.holding = True
        elif cmd.action == "put_down":
            if not self.holding: return dict(ok=False, reason="not holding anything")
            self.holding = False
        elif cmd.action == "stop":
            self.stopped = True
        self.log.append(cmd)
        return dict(ok=True, status=self.status())

    def emergency_stop(self):
        self.stopped = True

    def status(self):
        return dict(x=round(self.x, 3), y=round(self.y, 3), heading=round(self.heading, 1), speed=self.speed, holding=self.holding)


# ------------------------------------------------------------------ controller
class RobotController:
    def __init__(self, adapter, limits=LIMITS, lexicon=None):
        self.adapter, self.limits, self.lexicon = adapter, dict(limits), lexicon or {}
        self._times, self._lock, self._exec = collections.deque(), threading.Lock(), threading.Lock()
        self.audit = collections.deque(maxlen=1000)

    def _log(self, sid, text, result):
        self.audit.append(dict(t=time.time(), session=sid, text=(text or "")[:200], status=result["status"], reply=result["reply"]))
        return result

    def emergency_stop(self, sid="-"):
        """Always works: bypasses rate limits, pending confirmations and parsing."""
        self.adapter.emergency_stop()
        return self._log(sid, "STOP", dict(status="stopped", reply="Stopped.", command="stop"))

    def is_stop(self, text):
        return any(t in STOP_WORDS for t in apply_lexicon(tokenize(text), self.lexicon))

    def handle(self, text, session=None, from_translation=False, sid="-"):
        """One user utterance (already in English) -> result dict with status and reply."""
        toks = apply_lexicon(tokenize(text), self.lexicon)
        if any(t in STOP_WORDS for t in toks):
            if session is not None: session.pending = None
            return self.emergency_stop(sid)
        pend = getattr(session, "pending", None) if session is not None else None
        if pend and time.time() > pend["expires"]:
            session.pending = pend = None
        if pend and is_confirmation(text, self.lexicon):
            session.pending = None
            if toks[0] in NO:
                return self._log(sid, text, dict(status="cancelled", reply="Cancelled."))
            return self._run(pend["command"], sid, text)
        parsed = parse(text, self.lexicon)
        if parsed.clarify:
            return self._log(sid, text, dict(status="clarify", reply=parsed.clarify))
        if parsed.command is None:
            return dict(status="not_a_command", reply="")
        cmd = parsed.command
        ok, why = validate(cmd, self.limits)
        if not ok:
            return self._log(sid, text, dict(status="rejected", reply=f"I can't do that: {why}.", command=cmd.describe()))
        if needs_confirmation(cmd, from_translation, self.limits):
            if session is None:
                return self._log(sid, text, dict(status="rejected", reply="This command needs confirmation, which requires a session."))
            session.pending = dict(command=cmd, expires=time.time() + self.limits["confirm_ttl_s"])
            return self._log(sid, text, dict(status="needs_confirmation", command=cmd.describe(),
                             reply=f"I understood: {cmd.describe()}. Say 'yes' to confirm or 'no' to cancel."))
        return self._run(cmd, sid, text)

    def _run(self, cmd, sid, text):
        now = time.time()
        with self._lock:
            while self._times and now - self._times[0] > 10:
                self._times.popleft()
            if cmd.action != "stop" and len(self._times) >= self.limits["max_cmds_per_10s"]:
                return self._log(sid, text, dict(status="rejected", reply="Too many commands too quickly; please wait a few seconds."))
            self._times.append(now)
        ok, why = validate(cmd, self.limits)                       # re-validated immediately before acting
        if not ok:
            return self._log(sid, text, dict(status="rejected", reply=f"I can't do that: {why}."))
        with self._exec:                                           # one physical action at a time (stop never waits for this)
            res = self.adapter.execute(cmd)
        if not res.get("ok"):
            return self._log(sid, text, dict(status="rejected", reply=f"I can't do that: {res.get('reason')}.", command=cmd.describe()))
        st = res.get("status") or {}
        reply = (f"Done: {cmd.describe()}." if cmd.action != "status" else
                 f"Status: at ({st.get('x')}, {st.get('y')}) m, heading {st.get('heading')} degrees, speed {st.get('speed')} m/s, "
                 f"{'holding an object' if st.get('holding') else 'not holding anything'}.")
        return self._log(sid, text, dict(status="executed", reply=reply, command=cmd.describe(), robot=res.get("status")))
