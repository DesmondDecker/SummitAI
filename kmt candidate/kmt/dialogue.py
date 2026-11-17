"""Dialogue manager: routes each message using session context.

What 'context aware' means here (and does not): it tracks the conversation's language, remembered facts, the last
translation (for follow-ups like "and in Mende?"), and pending robot confirmations. It does NOT give the small
model deep reasoning; hard questions can be escalated to an optional large-model backend."""
import re
from .backends import BackendError
from .session import SessionManager
from . import robot as robotlib

_WH = {"what", "who", "where", "when", "why", "how", "which", "is", "are", "do", "does", "can", "should", "will"}
_ASKVERBS = {"explain", "tell", "describe", "list", "give", "help", "teach", "show", "compare", "summarize", "summarise",
             "suggest", "recommend", "solve", "plan", "define", "advise"}
_HARD = {"why", "explain", "compare", "plan", "recommend", "should", "difference", "advice", "solve", "analyze", "analyse", "strategy", "steps"}
_FOLLOW = re.compile(r"^(?:and |now |also |then )?(?:in|into|to) (krio|mende|temne|english)\s*[?.!]?$", re.I)
_REMEMBER = re.compile(r"^(?:remember|note)(?: that)?[:,]? (.+)$", re.I)
SYSTEM = ("You are a helpful assistant for speakers of Krio, Mende, Temne and English. Reply in the user's language "
          "when you can, otherwise in English. Answer directly in one or two short sentences unless asked for detail. "
          "Be correct, say when unsure, and never control physical devices.")


def is_complex(text):
    words = re.findall(r"\w+", text.lower())
    return len(words) > 40 or bool(_HARD & set(words)) or text.count("?") > 1


class Dialogue:
    def __init__(self, assistant, robot=None, backend=None, sessions=None, max_chars=2000):
        self.bot, self.robot, self.backend = assistant, robot, backend
        self.sessions, self.max_chars = sessions or SessionManager(), max_chars

    # ------------------------------------------------------------------
    def respond(self, text, sid="default", task="auto", lang=None, src=None, tgt=None):
        text = " ".join((text or "").split())[: self.max_chars]
        s = self.sessions.get(sid)
        if not text:
            return self._out(s, "", "empty")
        if self.robot and self.robot.is_stop(text):                       # emergency stop outranks everything
            s.pending = None
            r = self.robot.emergency_stop(sid); s.add("user", text); s.add("assistant", r["reply"])
            return self._out(s, r["reply"], "robot", source="rules", robot=r)
        s.add("user", text)
        lang = lang or self.bot.identify(text)[0]
        s.langs.append(lang)
        if task == "identify":
            best, probs = self.bot.identify(text)
            return self._out(s, best, "identify", lang=best, probabilities={k: round(v, 4) for k, v in probs.items()})
        if task == "translate":
            return self._translate(s, text, src or lang, tgt)
        if task == "ask":
            return self._ask(s, text, lang)
        if task == "robot":
            return self._robot(s, text, lang, force=True)
        if task == "remember":
            return self._remember(s, text)
        # ---- auto routing -------------------------------------------------
        m = _FOLLOW.match(text)
        if m and s.last_translation:                                       # "and in Mende?" re-translates the last source
            tgt_l = {"english": "en"}.get(m.group(1).lower(), m.group(1).lower())
            lt = s.last_translation
            return self._translate(s, lt["src"], lt["src_lang"], tgt_l, note="follow-up")
        m = _REMEMBER.match(text)
        if m:
            return self._remember(s, m.group(1))
        if self.robot:
            r = self._robot(s, text, lang)
            if r is not None:
                return r
        first = text.split()[0].lower().strip(",.!")
        if text.endswith("?") or first in _WH or first in _ASKVERBS:
            return self._ask(s, text, lang)
        if self.backend is not None and getattr(self.backend, "answer_all", False):
            return self._ask(s, text, lang)
        known = [l for l in s.langs if l != "en"]
        tgt_l = "en" if lang != "en" else (known[-1] if known else None)
        if tgt_l is None:
            return self._out(s, "Which language should I translate into? (krio, mende, temne)", "clarify", lang=lang)
        return self._translate(s, text, lang, tgt_l)

    # ------------------------------------------------------------------
    def _out(self, s, reply, task, **kw):
        if reply:
            s.add("assistant", reply)
        return dict(reply=reply, task=task, session=s.sid, turns=len(s.history), **kw)

    def _translate(self, s, text, src, tgt, note=None):
        tgt = tgt or ("en" if src != "en" else "krio")
        r = self.bot.translate(text, src, tgt)
        if r["source"] == "abstained":
            msg = "I can't translate that reliably yet."
            if r.get("suggestions"):
                msg += " Closest phrases I know: " + "; ".join(f"{x['source']} = {x['translation']}" for x in r["suggestions"][:3])
            return self._out(s, msg, "translate", lang=src, target=tgt, source="abstained", confidence=round(r["confidence"], 3),
                             suggestions=r.get("suggestions", []))
        s.last_translation = dict(src=text, src_lang=src, tgt=r["text"], tgt_lang=tgt)
        extra = {k: r[k] for k in ("confidence", "warning", "chunks", "alternatives", "match") if k in r}
        if r.get("alternatives"):
            r = dict(r, text=f"{r['text']} (also: {'; '.join(r['alternatives'])})")
        if r.get("match") == "near" and r.get("matched"):                  # never silent: say which entry was matched
            r = dict(r, text=f"{r['text']} (closest known entry: {r['matched']})")
        if note:
            extra["note"] = note
        return self._out(s, r["text"], "translate", lang=src, target=tgt, source=r["source"], **extra)

    def _remember(self, s, fact):
        new = s.add_fact(fact)
        return self._out(s, "Noted." if new else "I already have that.", "remember", source="session")

    def _ask(self, s, question, lang):
        facts = (s.recall(question, 2) + [t for _, it in self.bot.recall(question, 2) for t in [it["text"]]])[:3]
        local = None
        if facts and not is_complex(question):
            local = self.bot.ask(question, lang, context=" ".join(facts))
            if local.get("answer"):
                return self._out(s, local["answer"], "ask", lang=lang, source="local", confidence=round(local["confidence"], 3), support=facts)
        if self.backend is not None:                                        # too hard for the local model: escalate
            try:
                msgs = self._messages(s)
                reply = self.backend.complete(msgs, system=SYSTEM, max_tokens=96)
                return self._out(s, reply, "ask", lang=lang, source=f"backend:{self.backend.name}")
            except BackendError:
                w = "The larger model is unavailable right now."
                return self._out(s, "I can't answer that reliably right now.", "ask", lang=lang, source="none", warning=w)
        why = "no matching facts" if not facts else "low confidence or the question is too complex"
        return self._out(s, "I don't have enough information to answer that reliably.", "ask", lang=lang, source="none",
                         reason=why, escalation_available=False)

    def _messages(self, s, n=6):
        msgs = []
        for role, text in s.recent(n):
            role = "user" if role == "user" else "assistant"
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n" + text
            else:
                msgs.append(dict(role=role, content=text))
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    def _robot(self, s, text, lang, force=False):
        """Returns a result dict, or None if the text is not robot-related (auto mode only)."""
        if not self.robot:
            return self._out(s, "Robot control is not enabled on this server.", "robot", source="rules") if force else None
        from_tr, eng = False, text
        pend = s.pending
        if pend and robotlib.is_confirmation(text, self.robot.lexicon):      # yes/no must never depend on language detection
            r = self.robot.handle(text, s, sid=s.sid)
            return self._out(s, r["reply"], "robot", lang=lang, source="rules", robot=r)
        looks = robotlib.parse(text, self.robot.lexicon)
        if lang != "en" and not (looks.command or looks.clarify):          # non-English: translate, then ALWAYS confirm
            if pend or force or self._maybe_robot_words(text):
                tr = self.bot.translate(text, lang, "en")
                eng, from_tr = tr["text"], True
                if "confidence" in tr and tr["confidence"] < 0.5 and not pend:
                    return self._out(s, "I'm not sure I understood that command. Please rephrase it.", "robot", source="rules")
        if not pend and not force:
            p = robotlib.parse(eng, self.robot.lexicon)
            if p.command is None and not p.clarify:
                return None
        r = self.robot.handle(eng, s, from_translation=from_tr, sid=s.sid)
        if r["status"] == "not_a_command":
            return None if not force else self._out(s, "That doesn't look like a robot command.", "robot", source="rules")
        return self._out(s, r["reply"], "robot", lang=lang, source="rules", robot=r)

    def _maybe_robot_words(self, text):
        """Cheap gate so ordinary non-English chat is not sent through translation just to look for commands."""
        return len(text.split()) <= 12
