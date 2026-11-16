"""Run: python tests/test_all.py   (gradient check, tokenizer, memory)"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from kmt.transformer import Transformer
from kmt.tokenizer import BPE
from kmt.memory import MemoryStore
from kmt.autograd import set_dtype, cross_entropy, Tensor, embed, softmax


def test_transformer_gradients():
    rng = np.random.default_rng(3)
    m = Transformer(vocab_size=13, d=8, n_layer=2, n_head=2, ctx=8, dropout=0.0, seed=3)
    for p in m.params.values():                         # larger weights so every path matters
        p.data += rng.normal(0, 0.1, p.data.shape)
    ids = rng.integers(0, 13, (3, 7)); mask = (rng.random((3, 7)) > 0.3).astype(float); mask[:, 3] = 1
    m.loss(ids, mask, train=False).backward()
    worst = 0.0
    for name, p in m.params.items():
        for _ in range(3):
            ix = tuple(rng.integers(0, s) for s in p.data.shape)
            old = p.data[ix]; e = 1e-5
            p.data[ix] = old + e; lp = float(m.loss(ids, mask, train=False).data)
            p.data[ix] = old - e; lm = float(m.loss(ids, mask, train=False).data)
            p.data[ix] = old
            num, ana = (lp - lm) / (2 * e), p.grad[ix]
            worst = max(worst, abs(num - ana) / max(1e-7, abs(num) + abs(ana)))
    print("transformer worst relative grad error:", worst)
    assert worst < 1e-4


def test_causality():
    m = Transformer(vocab_size=11, d=8, n_layer=2, n_head=2, ctx=10, dropout=0.0, seed=1)
    a = np.array([[1, 2, 3, 4, 5, 6]]); b = a.copy(); b[0, 5] = 9
    la, lb = m.forward(a).data, m.forward(b).data
    assert np.allclose(la[0, :5], lb[0, :5])            # changing a later token must not affect earlier logits


def test_tokenizer_roundtrip():
    texts = ["Aw yu du? A dey fayn.", "Wetin na yu nem?", "Nɔ wɔri, God bless yu."] * 5
    tok = BPE.train(texts, 60, min_freq=2)
    for t in texts[:3]:
        assert tok.decode(tok.encode(t)) == t, t
    with tempfile.TemporaryDirectory() as d:
        tok.save(d + "/t.json"); assert BPE.load(d + "/t.json").encode(texts[0]) == tok.encode(texts[0])


def test_ascii_fold_matches_typed_special_letters():
    tok = BPE.train(["a tel yu God tenki", "aw yu du", "yu go tel am"] * 4, 30, min_freq=1, fold=True)
    assert tok.encode("a t\u025bl yu G\u0254d") == tok.encode("a tel yu God")          # typed with special letters == plain
    with tempfile.TemporaryDirectory() as d:
        tok.save(d + "/t.json"); t2 = BPE.load(d + "/t.json")
        assert t2.fold and t2.encode("t\u025bl") == tok.encode("tel")
    m = MemoryStore(None); m.add_pair("krio", "a tel yu", "en", "I tell you")
    assert m.search("a t\u025bl yu", 1, kind="translation", lang="krio", tgt_lang="en")[0][0] == 1.0       # memory also ignores the special letters


def test_memory():
    m = MemoryStore(None)
    m.add_pair("krio", "Aw yu du?", "en", "How are you?")
    m.add_pair("krio", "Usai yu dey go?", "en", "Where are you going?")
    s, it = m.search("aw yu du", 1, kind="translation", lang="krio", tgt_lang="en")[0]
    assert it["tgt"] == "How are you?" and s > 0.8


def _rand_model(seed=5):
    m = Transformer(vocab_size=15, d=16, n_layer=2, n_head=2, ctx=24, dropout=0.0, seed=seed)
    r = np.random.default_rng(seed)
    for p in m.params.values():
        p.data += r.normal(0, 0.15, p.data.shape)
    return m


def test_fast_inference_matches_training_forward():
    m = _rand_model(); ids = np.array([[1, 4, 7, 2, 9, 3, 5, 8]])
    ref = m.forward(ids).data
    full, _ = m.infer(ids, all_pos=True)
    assert np.allclose(ref, full, atol=1e-9)
    logits, cache = m.infer(ids[:, :5], all_pos=True)                # incremental with KV cache
    for t in range(5, 8):
        step, cache = m.infer(ids[:, t:t + 1], cache)
        assert np.allclose(step[0, 0], ref[0, t], atol=1e-9)


def test_cached_greedy_equals_uncached_and_beam1():
    m = _rand_model(); prefix = [1, 4, 7]; eos = 2
    ids = list(prefix)
    for _ in range(10):                                              # slow reference: recompute everything
        nxt = int(np.argmax(m.forward(np.array([ids])).data[0, -1])); ids.append(nxt)
        if nxt == eos: break
    fast, _ = m.generate(prefix, 10, eos)
    assert fast == ids[len(prefix):]
    b1, _ = m.beam_search(prefix, 10, eos, beam=1)
    assert b1 == fast
    b4, _ = m.beam_search(prefix, 10, eos, beam=4)
    assert len(b4) >= 1


def test_label_smoothing_gradient():
    r = np.random.default_rng(0); z = r.normal(size=(6, 9)); t = r.integers(0, 9, 6); w = r.random(6)
    L = Tensor(z, requires_grad=True); cross_entropy(L, t, w, smoothing=0.1).backward()
    num = np.zeros_like(z)
    for i in range(6):
        for j in range(9):
            zp, zm = z.copy(), z.copy(); zp[i, j] += 1e-6; zm[i, j] -= 1e-6
            num[i, j] = (float(cross_entropy(Tensor(zp), t, w, 0.1).data) - float(cross_entropy(Tensor(zm), t, w, 0.1).data)) / 2e-6
    assert np.abs(num - L.grad).max() < 1e-6


def test_float32_training_step():
    set_dtype("float32")
    try:
        m = Transformer(vocab_size=11, d=8, n_layer=1, n_head=2, ctx=8, dropout=0.1, seed=0)
        ids = np.random.default_rng(0).integers(0, 11, (4, 8)); mask = np.ones((4, 8))
        l = m.loss(ids, mask, smoothing=0.05); l.backward()
        assert m.params["tok"].data.dtype == np.float32 and m.params["tok"].grad.dtype == np.float32
        assert np.isfinite(float(l.data))
    finally:
        set_dtype("float64")


def test_embed_backward_matches_reference():
    r = np.random.default_rng(0); W = Tensor(r.normal(size=(20, 5)), requires_grad=True)
    idx = r.integers(0, 20, (4, 9)); out = embed(W, idx); g = r.normal(size=out.shape)
    out.grad = g; out._backward()
    ref = np.zeros((20, 5)); np.add.at(ref, idx, g)
    assert np.allclose(W.grad, ref)


def test_fused_softmax_gradient():
    r = np.random.default_rng(1); x = r.normal(size=(2, 2, 5, 5)); mask = np.triu(np.full((5, 5), -1e9), 1)
    w = r.normal(size=x.shape); X = Tensor(x, requires_grad=True)
    (softmax(X, mask, 0.7) * w).sum().backward()
    f = lambda a: float((softmax(Tensor(a), mask, 0.7).data * w).sum())
    num = np.zeros_like(x)
    for i in np.ndindex(*x.shape):
        a, b = x.copy(), x.copy(); a[i] += 1e-6; b[i] -= 1e-6; num[i] = (f(a) - f(b)) / 2e-6
    assert np.abs(num - X.grad).max() < 1e-6


def test_selective_loss_equals_full_loss():
    m = _rand_model(2); r = np.random.default_rng(2)
    ids = r.integers(0, 15, (3, 9)); mask = (r.random((3, 9)) > 0.4).astype(float); mask[0, 4] = 1
    logits = m.forward(ids[:, :-1]); B, T, V = logits.shape
    full = cross_entropy(logits.reshape(B * T, V), ids[:, 1:].reshape(-1), mask[:, 1:].reshape(-1))
    assert abs(float(full.data) - float(m.loss(ids, mask, train=False).data)) < 1e-10


def test_graph_memory_is_freed_after_backward():
    """Regression: backward closures formed reference cycles, so graphs piled up between rare garbage collections (training was killed at ~4 GB)."""
    import gc, weakref
    gc.collect(); gc.disable()
    try:
        x = Tensor(np.ones((3, 3)), requires_grad=True)
        y = (x * 2.0).tanh(); z = y.sum(); ref = weakref.ref(y)
        z.backward(); del y, z
        assert ref() is None, "intermediate tensor survived: the graph is still referenced"
        assert x.grad is not None
    finally:
        gc.enable()


def test_no_float64_leak_in_float32_mode():
    set_dtype("float32")
    try:
        m = Transformer(vocab_size=11, d=8, n_layer=2, n_head=2, ctx=12, dropout=0.1, seed=0)
        ids = np.random.default_rng(0).integers(0, 11, (3, 10)); mask = np.ones((3, 10))
        loss = m.loss(ids, mask, smoothing=0.05)
        seen, stack, bad = set(), [loss], []
        while stack:
            n = stack.pop()
            if id(n) in seen: continue
            seen.add(id(n)); stack.extend(n._parents)
            if n.data.dtype != np.float32 and n.data.ndim > 0: bad.append(n.data.shape)
        loss.backward()
        for n in m.parameters():
            if n.grad is not None and n.grad.dtype != np.float32: bad.append(("grad", n.data.shape))
        assert not bad, f"float64 leaked into {len(bad)} tensors, e.g. {bad[:3]}"
        lg, _ = m.infer(ids[:, :5], all_pos=True); assert lg.dtype == np.float32
    finally:
        set_dtype("float64")


def _tiny_assistant(tmp):
    """A small random (untrained) model + tokenizer on disk: enough to test robustness, not accuracy."""
    from kmt.assistant import Assistant
    tok = BPE.train(["aw yu du a dey fayn tenki the big dog eats bread"] * 4, 40, min_freq=1)
    m = Transformer(len(tok), d=16, n_layer=1, n_head=2, ctx=48, dropout=0.0, seed=0)
    os.makedirs(tmp + "/ck"); m.save(tmp + "/ck/kmt.npz", {"langs": ["krio", "en"], "intents": ["greeting", "thanks"]})
    tok.save(tmp + "/ck/tokenizer.json")
    return Assistant(tmp + "/ck", tmp + "/mem", tmp + "/nodata")


def test_assistant_survives_hostile_input():
    with tempfile.TemporaryDirectory() as tmp:
        b = _tiny_assistant(tmp)
        for text in ["", "   ", "\n\t", "?", "\U0001F600" * 3, "\u00e9" * 500, "x" * 2000, "dog " * 3000, "a"]:
            b.identify(text); b.translate(text, "krio", "en", use_memory=False); b.predict_next(text, "en")
            b.classify_intent(text, "en"); b.score(text or "x", "en"); b.generate("krio", text[:50])
        long = b.translate("dog " * 400, "krio", "en", use_memory=False)
        assert long["chunks"] > 1                                     # split, not rejected, not recursing forever
        for giant in ("\u00e9" * 500, "x" * 2000):                     # one unsplittable "word" must still be processed
            r = b.translate(giant, "krio", "en", use_memory=False)
            assert r["source"] == "model" and r["chunks"] > 1, r


def test_assistant_memory_persists_and_newest_correction_wins():
    from kmt.assistant import Assistant
    with tempfile.TemporaryDirectory() as tmp:
        b = _tiny_assistant(tmp)
        b.correct_translation("krio", "aw yu du", "en", "first"); b.correct_translation("krio", "aw yu du", "en", "second")
        with open(tmp + "/mem/memory.jsonl", "a") as f:
            f.write('{"kind": "fact", "text": "trunc')                 # a half-written line must not break loading
        r = Assistant(tmp + "/ck", tmp + "/mem", tmp + "/nodata").translate("aw yu du", "krio", "en")
        assert r["source"] == "memory" and r["text"] == "second"


def test_assistant_abstains_when_calibration_says_untrusted():
    import json
    from kmt.assistant import Assistant
    with tempfile.TemporaryDirectory() as tmp:
        b = _tiny_assistant(tmp)
        r = b.translate("dog dog dog", "krio", "en", use_memory=False)
        assert r["source"] == "model"                                           # no calibration file: behaves as before
        json.dump({"default": 1.01, "directions": {}}, open(tmp + "/ck/calibration.json", "w"))
        b = Assistant(tmp + "/ck", tmp + "/mem", tmp + "/nodata")
        r = b.translate("dog dog dog", "krio", "en", use_memory=False)
        assert r["source"] == "abstained" and r["text"] == "" and "reason" in r   # refuses instead of guessing
        assert b.translate("dog dog dog", "krio", "en", use_memory=False, force=True)["source"] == "model"
        b.correct_translation("krio", "dog dog dog", "en", "known phrase")
        r = b.translate("dog dog dog", "krio", "en")
        assert r["source"] == "memory" and r["text"] == "known phrase"          # known phrasebook entries still answer


def test_lookup_canonical_keys_alternatives_and_quality():
    m = MemoryStore(None)
    m.add_pair("krio", "Just now.", "en", "Right now", q=0.6)
    m.add_pair("krio", "just now", "en", "At this moment", q=0.9)
    lk = m.lookup("JUST NOW!", "krio", "en")
    assert lk["status"] == "exact" and lk["answer"] == "At this moment" and "Right now" in lk["alternatives"]   # best quality first, both kept


def test_lookup_did_you_mean_and_short_word_rules():
    m = MemoryStore(None)
    for s_, t_ in [("gboto", "one"), ("gbori", "two"), ("kpenge", "three"), ("wuloma", "four")]:
        m.add_pair("mende", s_, "en", t_)
    for auto in (False, True):
        lk = m.lookup("gboti", "mende", "en", short_auto=auto)               # two neighbours at the same distance: never guess
        assert lk["status"] == "suggest" and {x["translation"] for x in lk["suggestions"]} >= {"one", "two"}
    assert m.lookup("kpengee", "mende", "en", short_auto=True)["answer"] == "three"      # one unambiguous neighbour
    assert m.lookup("kpengee", "mende", "en")["status"] == "suggest"                      # default: offered, not asserted
    assert m.lookup("qqqqqq", "mende", "en")["status"] == "none"


def test_seeding_skips_low_quality_entries():
    from kmt.assistant import Assistant
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_assistant(tmp); os.makedirs(tmp + "/data/parallel")
        open(tmp + "/data/parallel/krio-en.tsv", "w", encoding="utf-8").write(
            "# krio\tenglish\tsource\tquality\naw yu du\tHow are you\tcurated\t0.95\nbad ocr entry\tgarbage\tocr\t0.30\n")
        b = Assistant(tmp + "/ck", tmp + "/mem2", tmp + "/data")
        assert b.memory.lookup("aw yu du", "krio", "en")["status"] == "exact"
        assert b.memory.lookup("bad ocr entry", "krio", "en")["status"] != "exact"       # withheld until a human confirms it


def test_gloss_lookup_from_dictionary_entries():
    with tempfile.TemporaryDirectory() as tmp:
        b = _tiny_assistant(tmp)
        b.memory.add_pair("krio", "dog", "en", "canine"); b.memory.add_pair("krio", "yu du", "en", "hello")   # 2nd is a phrase: not a lexicon word
        g = b.gloss("Dog yu zzz", "krio")
        assert g["words"] == [("Dog", "canine"), ("yu", None), ("zzz", None)] and abs(g["coverage"] - 1 / 3) < 1e-9
        assert b.gloss("", "krio") == {"words": [], "coverage": 0.0}


def test_assistant_detects_mismatched_files():
    from kmt.assistant import Assistant
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_assistant(tmp)
        BPE.train(["just a few words words"], 3, min_freq=1).save(tmp + "/ck/tokenizer.json")
        for args, exc in (((tmp + "/ck", tmp + "/m", tmp + "/d"), ValueError), ((tmp + "/none",), FileNotFoundError)):
            try:
                Assistant(*args); raise AssertionError("should have failed")
            except exc:
                pass


def test_checkpoint_roundtrip_is_exact_and_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        m = _rand_model(3); m.save(tmp + "/a.npz", {"k": 1})
        m2, extra = Transformer.load(tmp + "/a.npz")
        ids = np.array([[1, 2, 3, 4]])
        assert extra == {"k": 1} and np.array_equal(m.forward(ids).data, m2.forward(ids).data)
        assert not [f for f in os.listdir(tmp) if "tmp" in f]        # no leftover temp file


def test_memory_scales_and_matches_exactly():
    m = MemoryStore(None); r = np.random.default_rng(0)
    words = ["".join(r.choice(list("abdegiklmnoprstu"), r.integers(2, 7))) for _ in range(500)]
    for _ in range(3000):
        m.add_pair("krio", " ".join(r.choice(words, 8)), "en", " ".join(r.choice(words, 8)))
    it = m.items[1234]
    hit = m.search(it["src"], 1, kind="translation", lang=it["lang"], tgt_lang=it["tgt_lang"])[0]
    assert hit[0] == 1.0 and hit[1] == it
    fuzzy = m.search(it["src"] + " zz", 1, kind="translation", lang=it["lang"], tgt_lang=it["tgt_lang"])[0]
    assert fuzzy[1]["src"] == it["src"] and fuzzy[0] > 0.8
    assert m.search("anything", 3, kind="fact") == []                # filters respected


def test_distillation_rejects_unverified_teacher_steps():
    from kmt.distillation import OllamaTeacher, make_expressions, validate_solution
    from tools.teach import _expected_record, _read_pairs
    from unittest import mock
    expressions = make_expressions(12, 19)
    assert expressions == make_expressions(12, 19) and len(set(expressions)) == 12
    expression = "(8 + 4) * 3"
    solution = _expected_record(expression)
    assert validate_solution(expression, solution) == solution
    bad = dict(solution, steps=["9 + 3 = 12", solution["steps"][1]])
    try:
        validate_solution(expression, bad)
        raise AssertionError("unrelated teacher steps were accepted")
    except ValueError:
        pass
    teacher = OllamaTeacher()
    attempts = []

    def intermittent_failure(expressions):
        attempts.append(expressions)
        if len(attempts) < 3:
            raise RuntimeError("temporary local teacher timeout")
        return [_expected_record(expression) for expression in expressions]

    teacher._solve_batch = intermittent_failure
    with mock.patch("kmt.distillation.time.sleep"):
        assert teacher.solve([expression]) == [_expected_record(expression)]
    assert len(attempts) == 3
    try:
        OllamaTeacher(url="http://example.com")
        raise AssertionError("remote teacher endpoint was accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.tsv")
        with open(path, "w", encoding="utf-8") as out:
            out.write("safe\tEnglish\tmigeod1908:phrases\t0.60\n")
            out.write("restricted\tEnglish\tjw_krio\t1.00\n")
            out.write("low quality\tEnglish\tpeacecorps_mende\t0.30\n")
        assert _read_pairs(path) == [("safe", "English")]


def test_qwen_data_splits_and_response_only_mask():
    from tools.prepare_qwen_data import (_valid_message, conversation_examples, local_translation_examples,
                                        split_examples)
    from tools.train_qwen_lora import encode_example

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            ids = [1]
            for message in messages:
                ids.append({"system": 2, "user": 3, "assistant": 4}[message["role"]])
                ids.extend([10 + ord(char) % 50 for char in message["content"]])
            if add_generation_prompt:
                ids.append(4)
            return ids

    encoded = encode_example(FakeTokenizer(), {"messages": [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
    ]}, max_length=1000)
    first_target = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
    assert first_target > 0 and all(label == -100 for label in encoded["labels"][:first_target])
    accepted = {"lang": "en", "role": "assistant", "text": "answer", "tree_state": "ready_for_export",
                "labels": {"spam": {"value": 0}, "pii": {"value": 0}}}
    assert _valid_message(accepted)
    assert not _valid_message({**accepted, "tree_state": "halted_by_moderator"})
    assert not _valid_message({**accepted, "labels": {"pii": {"value": 1}}})
    tree = [
        {"message_id": "u", "message_tree_id": "tree", "parent_id": None, "role": "prompter",
         "lang": "en", "text": "Question one", "_hf_split": "train"},
        {"message_id": "a", "message_tree_id": "tree", "parent_id": "u", "role": "assistant",
         "lang": "en", "text": "Answer one", "_hf_split": "train"},
        {"message_id": "u2", "message_tree_id": "tree", "parent_id": "a", "role": "prompter",
         "lang": "en", "text": "Question two", "_hf_split": "validation"},
        {"message_id": "a2", "message_tree_id": "tree", "parent_id": "u2", "role": "assistant",
         "lang": "en", "text": "Answer two", "_hf_split": "validation"},
    ]
    oasst_splits = split_examples(conversation_examples(tree))
    assert len(oasst_splits["test"]) == 2 and not oasst_splits["train"] and not oasst_splits["validation"]

    with tempfile.TemporaryDirectory() as tmp:
        for split in ("parallel", "heldout"):
            os.makedirs(f"{tmp}/{split}")
        open(f"{tmp}/parallel/krio-en.tsv", "w", encoding="utf-8").write(
            "a yu\tHow are you?\tpeacecorps_krio\t0.60\n"
            "restricted\tbad\tjw_krio\t1.00\n"
            "low\tquality\tpeacecorps_mende\t0.30\n")
        open(f"{tmp}/heldout/krio-en.tsv", "w", encoding="utf-8").write(
            "aw yu du?\tHow are you?\tpeacecorps_krio\t0.60\n")
        examples, counts = local_translation_examples(tmp)
        assert counts == {"train": 2, "test": 2}
        assert all("restricted" not in str(row) for row in examples)
        splits = split_examples(examples)
        assert len(splits["train"]) == 2 and len(splits["test"]) == 2 and not splits["validation"]


def test_qwen_local_runtime_and_distillation_gates():
    from kmt.qwen_runtime import DEFAULT_SYSTEM, normalize_messages, trim_history
    from tools.distill_qwen import teacher_answer
    from tools.evaluate_qwen import token_f1
    from tools.train_qwen_lora import load_approved_teacher_rows
    import json

    turns = normalize_messages([{"role": "user", "content": "Hi"}], None)
    assert turns[0] == {"role": "system", "content": DEFAULT_SYSTEM}
    trimmed = trim_history([turns[0], {"role": "user", "content": "old"},
                            {"role": "assistant", "content": "old answer"},
                            {"role": "user", "content": "latest"}], lambda rows: len(rows), 3)
    assert [row["role"] for row in trimmed] == ["system", "user"]
    assert token_f1("a b c", "b c d") == 2 / 3
    try:
        teacher_answer([{"role": "user", "content": "hi"}], "model", "https://example.com")
        raise AssertionError("remote teacher endpoint was accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "teacher.jsonl")
        rows = [
            {"review_status": "pending", "messages": []},
            {"review_status": "approved", "messages": [
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        ]
        with open(path, "w", encoding="utf-8") as out:
            out.write("\n".join(json.dumps(row) for row in rows) + "\n")
        assert len(load_approved_teacher_rows(path, 1)) == 1
        assert load_approved_teacher_rows(path, 0) == []


def test_qwen_verified_arithmetic_curriculum_and_math_eval():
    import json
    from kmt.distillation import canonical_solution, make_expressions
    from tools.generate_verified_arithmetic import EVAL_COUNT, EVAL_SEED, generate_rows
    from tools.evaluate_qwen import ARITHMETIC_REGRESSION_ANSWER, evaluate_arithmetic
    from tools.qwen_training_loop import _publish_adapter, promotion_decision
    from tools.train_qwen_lora import estimate_parameter_count, load_approved_teacher_rows
    from kmt.qwen_runtime import resolve_adapter_path
    from types import SimpleNamespace

    class Teacher:
        model = "local-test-teacher"

        def solve(self, expressions):
            return [canonical_solution(expression) for expression in expressions]

    reserved = set(make_expressions(EVAL_COUNT, EVAL_SEED))
    rows = generate_rows(Teacher(), 5, 123, reserved, set())
    assert len(rows) == 5
    assert all(row["review_status"] == "approved" for row in rows)
    assert all(row["verification"]["expression"] not in reserved for row in rows)
    assert len({row["group_id"] for row in rows}) == 5

    class PerfectRuntime:
        def respond(self, messages, max_new_tokens):
            expression = messages[0]["content"].split("Evaluate exactly: ", 1)[1].splitlines()[0]
            if expression == "67897644457 * 1235656768979":
                return f"Answer: {ARITHMETIC_REGRESSION_ANSWER}"
            solution = canonical_solution(expression)
            steps = "\n".join(
                f"Step {index}: {step}" for index, step in enumerate(solution["steps"], 1)
            )
            return f"{steps}\nAnswer: {solution['answer']}"

    scores = evaluate_arithmetic(PerfectRuntime(), 4)
    assert scores["examples"] == 5
    assert scores["exact_answer_accuracy"] == 1
    assert scores["valid_step_accuracy"] == 1
    assert scores["large_number_regression"]["correct"]

    config = SimpleNamespace(
        hidden_size=2560, num_hidden_layers=36, vocab_size=151936,
        intermediate_size=9728, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, tie_word_embeddings=True,
    )
    assert estimate_parameter_count(config) > 3_000_000_000

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "teacher.jsonl")
        unsafe_rows = [
            *rows,
            {"review_status": "approved", "messages": ["malformed"]},
            {"review_status": "pending", "messages": [
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        ]
        with open(path, "w", encoding="utf-8") as target:
            target.writelines(json.dumps(row) + "\n" for row in unsafe_rows)
        assert len(load_approved_teacher_rows(path, 20)) == 5

    before = {
        "arithmetic_reasoning": {
            "exact_answer_accuracy": 0.5,
            "valid_step_accuracy": 0.4,
            "large_number_regression": {"correct": False},
        },
        "heldout_conversation": {"token_f1": 0.7},
        "translation_chrF": {"krio": {"chrF": 0.2}},
    }
    improved = {
        "arithmetic_reasoning": {
            "exact_answer_accuracy": 0.6,
            "valid_step_accuracy": 0.4,
            "large_number_regression": {"correct": True},
        },
        "heldout_conversation": {"token_f1": 0.69},
        "translation_chrF": {"krio": {"chrF": 0.19}},
    }
    assert promotion_decision(before, improved) == (True, [])
    regressed = {**improved, "translation_chrF": {"krio": {"chrF": 0.1}}}
    accepted, reasons = promotion_decision(before, regressed)
    assert not accepted and any("translation" in reason for reason in reasons)

    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "student")
        adapter = os.path.join(output, "candidates", "cycle-0001", "adapter")
        os.makedirs(adapter)
        args = SimpleNamespace(
            base_model_id="Qwen/Qwen3-4B", base_revision="rev",
            base_license="Apache-2.0", parameter_count=4022468096,
        )
        _publish_adapter(output, adapter, args, improved, 1)
        assert resolve_adapter_path(output) == os.path.realpath(adapter)
        with open(os.path.join(output, "active.json"), encoding="utf-8") as source:
            pointer = json.load(source)
        pointer["adapter"] = "..\\outside"
        with open(os.path.join(output, "active.json"), "w", encoding="utf-8") as target:
            json.dump(pointer, target)
        try:
            resolve_adapter_path(output)
            raise AssertionError("adapter pointer escaped its model directory")
        except ValueError:
            pass


def test_bible_alignment_tool():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        books = ["GEN", "RUT", "JON", "MAT"]
        for lang in ("en", "kr"):
            os.makedirs(f"{tmp}/{lang}")
            for b in books:
                body = "\n".join(f"\\v {v} {'Word ' * (3 + v)}{'x' if lang == 'kr' else ''}. \\f + \\ft note \\f*" for v in range(1, 40))
                open(f"{tmp}/{lang}/{b}.usfm", "w", encoding="utf-8-sig").write(f"\\id {b}\n\\c 1\n\\s1 Heading\n" + body + "\n")
        tool = os.path.join(os.path.dirname(__file__), "..", "tools", "bible_align.py")
        out = subprocess.run([sys.executable, tool, "--lang", "krio", "--src", tmp + "/kr", "--english", tmp + "/en",
                              "--data", tmp + "/d", "--holdout", "RUT"], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        train = open(tmp + "/d/parallel/krio-en.tsv", encoding="utf-8").read()
        held = open(tmp + "/d/heldout/krio-en.tsv", encoding="utf-8").read()
        assert "note" not in train and "Heading" not in train         # footnotes and headings stripped
        assert train.count("\n") - 1 == 3 * 39 and held.count("\n") - 1 == 39   # whole book held out, rest kept
        # the trainer's Dataset must read the held-out book as validation only, counted once, never in training
        from kmt.data import Dataset, collect_texts
        tok = BPE.train(collect_texts(tmp + "/d"), 30, min_freq=1)
        ds = Dataset(tmp + "/d", tok, 128)
        assert {g: len(v) for g, v in ds.val_by_group.items() if g.startswith("tr:")} == {"tr:krio>en": 39, "tr:en>krio": 39}


if __name__ == "__main__":
    for f in (test_transformer_gradients, test_causality, test_tokenizer_roundtrip, test_ascii_fold_matches_typed_special_letters, test_memory,
              test_fast_inference_matches_training_forward, test_cached_greedy_equals_uncached_and_beam1,
              test_label_smoothing_gradient, test_float32_training_step, test_embed_backward_matches_reference,
              test_fused_softmax_gradient, test_selective_loss_equals_full_loss, test_graph_memory_is_freed_after_backward, test_no_float64_leak_in_float32_mode,
              test_assistant_survives_hostile_input, test_assistant_memory_persists_and_newest_correction_wins,
              test_assistant_abstains_when_calibration_says_untrusted, test_lookup_canonical_keys_alternatives_and_quality, test_lookup_did_you_mean_and_short_word_rules, test_seeding_skips_low_quality_entries, test_gloss_lookup_from_dictionary_entries, test_assistant_detects_mismatched_files, test_checkpoint_roundtrip_is_exact_and_atomic,
              test_memory_scales_and_matches_exactly, test_distillation_rejects_unverified_teacher_steps,
              test_qwen_data_splits_and_response_only_mask, test_qwen_local_runtime_and_distillation_gates,
              test_qwen_verified_arithmetic_curriculum_and_math_eval,
              test_bible_alignment_tool):
        f(); print("passed", f.__name__)
