"""Interactive assistant.   python chat.py
Type plain text: non-English is translated to English, English to Krio (change with --to).
Commands: /tr krio>en TEXT   /id TEXT   /intent TEXT   /gen LANG [PROMPT]
          /ask QUESTION   /solve ARITHMETIC (experimental)   /next TEXT   /score TEXT   /gloss TEXT (experimental word lookup)
          /fix CORRECT TRANSLATION   /remember FACT   /recall QUERY   /help   /quit"""
import argparse
from kmt.assistant import Assistant

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints")
ap.add_argument("--to", default="krio", help="target language for English input")
a = ap.parse_args()
bot = Assistant(a.ckpt)
print(f"KMT ready (languages with data: {', '.join(bot.langs)}). /help for commands.")
last = None


def show(r):
    global last
    last = r
    extra = f"  [{r['source']}" + (f", conf {r['confidence']:.2f}" if "confidence" in r else "") + "]"
    if r["source"] == "abstained":
        print(f"{r['src_lang']}>{r['tgt_lang']}: no reliable translation available{extra}")
        for suggestion in r.get("suggestions", []):
            if suggestion["similarity"] < 0.6:
                continue
            print(f"  Phrasebook suggestion: {suggestion['source']} -> {suggestion['translation']}")
        print("  Use /id TEXT to inspect language detection, or /tr SOURCE>TARGET TEXT to choose a direction.")
    else:
        print(f"{r['src_lang']}>{r['tgt_lang']}: {r['text']}{extra}")
    if "warning" in r:
        print("  warning:", r["warning"])


while True:
    try:
        line = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue
    if line in ("/quit", "/exit"):
        break
    if line in ("/help", "help"):
        print(__doc__); continue
    cmd, _, rest = line.partition(" ")
    if not line.startswith("/"):
        if line.lower() in bot.langs:
            print(f"'{line}' is a language code, not text. Use /tr SOURCE>TARGET TEXT.")
            continue
        text = "hello" if line.lower() == "hi" else line
        if text == "hello" and bot.memory.lookup(text, "en", a.to)["status"] == "exact":
            r = bot.translate(text, "en", a.to)
        else:
            src, _ = bot.identify(text)
            r = bot.translate(text, src, "en" if src != "en" else a.to)
        r["orig"] = line; show(r)
    elif cmd == "/tr":
        pair, _, text = rest.partition(" ")
        s, _, t = pair.partition(">")
        r = bot.translate(text, s or None, t or "en"); r["orig"] = text; show(r)
    elif cmd == "/id":
        best, pr = bot.identify(rest); print(best, {k: round(v, 3) for k, v in pr.items()})
    elif cmd == "/intent":
        best, pr = bot.classify_intent(rest); print(best, {k: round(v, 3) for k, v in pr.items()})
    elif cmd == "/gen":
        lang, _, prompt = rest.partition(" "); print(bot.generate(lang, prompt))
    elif cmd == "/gloss":
        g = bot.gloss(rest, bot.identify(rest)[0])
        print(" ".join(f"{w}={x or '?'}" for w, x in g["words"]), f"  [covers {g['coverage']:.0%}; rough reading aid, NOT a translation]")
    elif cmd == "/ask":
        r = bot.ask(rest)
        print(r["answer"] if r.get("answer") else "I don't know" + (f" (best guess: {r['guess']}, conf {r['confidence']:.2f})" if r.get("guess") else ""))
        for f in r.get("support", []): print("  based on:", f)
    elif cmd == "/solve":
        r = bot.ask("Show the arithmetic steps and final answer.", lang="en",
                    context=f"Evaluate exactly: {rest}", min_conf=0.0, extractive=False, max_new=48)
        if r.get("guess"):
            print(f"{r['guess']}  [experimental, conf {r['confidence']:.2f}]")
        else:
            print("I couldn't solve that reliably.")
    elif cmd == "/next":
        print("  ".join(f"{t.strip() or '␣'} {p:.0%}" for t, p in bot.predict_next(rest, bot.identify(rest)[0])))
    elif cmd == "/score":
        r = bot.score(rest); print(f"perplexity {r['perplexity']:.1f}; least expected:", [(t, round(p, 3)) for t, p in sorted(r['tokens'], key=lambda x: x[1])[:3]])
    elif cmd == "/fix":
        if not last or "orig" not in last:
            print("Nothing to correct yet."); continue
        bot.correct_translation(last["src_lang"], last["orig"], last["tgt_lang"], rest)
        print("Thanks: stored in memory now, and queued for the next fine-tune (python finetune.py).")
    elif cmd == "/remember":
        print("Remembered." if bot.remember(rest) else "Already known.")
    elif cmd == "/recall":
        for s, it in bot.recall(rest):
            print(f"{s:.2f}  {it['text']}")
    else:
        print("Unknown command. /help")
