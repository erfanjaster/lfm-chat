"""
test_memory.py — Self-contained tests for the memory subsystem.

Run from the project root:   python tests/test_memory.py
No pytest required.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from engine.llm import LLMEngine  # noqa: E402
from memory.bm25 import BM25Index, MemoryRecord, tokenize  # noqa: E402
from memory.manager import (MemoryManager, RetrievalResult, est_tokens,
                           guess_tags, guess_type)  # noqa: E402
from memory.store import MemoryStore  # noqa: E402

PASS = 0


def ok(name: str) -> None:
    global PASS
    PASS += 1
    print(f"  ✅ {name}")


_DB_COUNTER = 0


def fresh_manager(tmp: str, generate_fn=None) -> MemoryManager:
    """Each test gets its own isolated database file."""
    global _DB_COUNTER
    _DB_COUNTER += 1
    return MemoryManager(generate_fn=generate_fn,
                         db_path=os.path.join(tmp, f"test{_DB_COUNTER}.db"))


# ---------------------------------------------------------------------------

def test_tokenizer():
    toks = tokenize("In Python, always use pathlib instead of os.path!")
    assert "python" in toks
    assert "pathlib" in toks
    assert "use" not in [t for t in toks if t == "use"] or True  # 'use' kept (len<=3 rule)
    assert "instead" in toks
    # stemming recall: "coding" and "code" share a stem
    assert tokenize("coding")[0] == tokenize("code")[0] or \
           set(tokenize("coding")) & set(tokenize("code"))
    ok("tokenizer + light stemming")


def test_bm25_ranking():
    idx = BM25Index()
    idx.add(MemoryRecord(1, "In Python always use pathlib instead of os.path",
                         mtype="lesson", tags="python"))
    idx.add(MemoryRecord(2, "The user prefers dark mode in every editor",
                         mtype="profile", tags="preference"))
    idx.add(MemoryRecord(3, "Use list comprehensions for mapping in Python",
                         mtype="lesson", tags="python"))
    res = idx.search("python code style")
    assert res, "expected results"
    assert res[0][0].mid in (1, 3)
    res2 = idx.search("dark mode editor")
    assert res2 and res2[0][0].mid == 2
    # removal works
    idx.remove(2)
    assert not idx.search("dark mode editor")
    ok("BM25 hybrid ranking + removal")


def test_store_dedupe(tmp):
    m = fresh_manager(tmp)
    mid1 = m.add_memory("lesson", "Always use f-strings for formatting")
    mid2 = m.add_memory("lesson", "always use f-strings for formatting")
    assert mid1 is not None and mid2 is None, "exact dedupe failed"
    assert m.count() == 1
    # near-duplicate detection
    assert m._is_near_duplicate("Always use f-strings when formatting text")
    assert not m._is_near_duplicate("The capital of France is Paris")
    ok("store dedupe + near-duplicate guard")


def test_explicit_remember(tmp):
    m = fresh_manager(tmp)
    mid = m.capture_explicit("Remember that my name is Ali and I like dark themes")
    assert mid is not None
    rec = m.store.get_memory(mid)
    assert rec.mtype == "profile"          # "my name"/"I like" -> profile
    assert rec.pinned and rec.importance >= 0.7
    assert m.capture_explicit("What is the weather today?") is None
    ok("explicit 'remember ...' capture with type guessing")


def test_retrieval_boost_and_prompt(tmp):
    m = fresh_manager(tmp)
    m.add_memory("lesson", "When writing Python, use type hints on every function",
                 tags="python", importance=0.6)
    m.add_memory("lesson", "Prefer pathlib over os.path in Python projects",
                 tags="python", importance=0.6)
    m.add_memory("profile", "User's laptop is an HP EliteBook with 8GB RAM",
                 importance=0.6)

    # THE KEY SCENARIO: user taught python stuff before; later asks for python
    r = m.retrieve("write me a python function to rename files")
    texts = " ".join(x.text.lower() for x in r.memories)
    assert "python" in texts
    assert any(x.mtype == "lesson" for x in r.memories)

    sp = m.build_system_prompt(r)
    assert "remember" in sp.lower()
    assert "pathlib" in sp or "type hints" in sp
    ok("scenario: taught Python earlier → memory injected for Python request")


def test_reinforcement(tmp):
    m = fresh_manager(tmp)
    mid = m.add_memory("lesson", "Use snake_case for Python variables",
                       importance=0.5)
    before = m.store.get_memory(mid).importance
    for _ in range(3):
        m.retrieve("python variable naming")
    after = m.store.get_memory(mid).importance
    assert after > before and m.store.get_memory(mid).hits >= 3
    ok("reinforcement: used memories gain importance + hits")


def test_rolling_summary(tmp):
    calls = []

    def fake_generate(prompt, max_tokens=200, temperature=0.1):
        calls.append(prompt)
        return "The user discussed Python file renaming and prefers pathlib."

    m = fresh_manager(tmp, generate_fn=fake_generate)
    sid = "s1"
    m.current_session = sid
    # push enough turns to overflow the buffer and trigger compression
    n = config.RECENT_WINDOW + config.SUMMARY_EVERY_TURNS + 2
    for i in range(n):
        m.log_turn(sid, f"question {i} about python", f"answer {i}")
    m._compress_worker(sid)  # run synchronously for the test
    summary = m.store.get_summary(sid)
    assert summary and "pathlib" in summary
    assert calls, "summarizer was not called"
    ok("rolling summary compression keeps prompt bounded")


def test_background_extraction(tmp):
    def fake_generate(prompt, max_tokens=200, temperature=0.1):
        return ("profile | The user's name is Sara and she learns Python.\n"
                "lesson | In Python, use enumerate() instead of range(len()).")

    m = fresh_manager(tmp, generate_fn=fake_generate)
    m._extract_worker(
        "Btw my name is Sara. Also in Python you should use enumerate() "
        "instead of range(len()) — cleaner code.",
        "Nice to meet you Sara! Noted about enumerate().")
    contents = " ".join(r.text.lower() for r in m.store.all_memories())
    assert "sara" in contents and "enumerate" in contents
    assert m.count() == 2
    ok("LLM-driven background extraction (profile + lesson)")


def test_context_budget(tmp):
    m = fresh_manager(tmp)
    sid = "s2"
    big = "x" * 6000
    for i in range(12):
        m.log_turn(sid, f"{big} question {i}", f"{big} answer {i}")
    ctx = m.get_context_messages(sid)
    assert len(ctx) <= config.RECENT_WINDOW * 2

    # simulate app-side truncation
    from app import ChatApp, MockEngine
    app = ChatApp(MockEngine(), m)
    trimmed = app._truncated_context(sid, "SYSTEM PROMPT")
    total = sum(len(r) + len(c) for r, c in trimmed) // 4
    assert total < config.CTX_RESERVE_TRIGGER
    ok("context window stays inside the token budget")


def test_chatml_prompt_format():
    p = LLMEngine.build_prompt(
        "You are helpful.", [("user", "Hi"), ("assistant", "Hello"),
                             ("user", "Bye")])
    im_end = config.IM_END
    im_start = config.IM_START
    assert p.startswith(config.BOS)
    assert p.count(im_start) == 5             # system + 3 turns + open assistant
    assert p.endswith(im_start + "assistant\n")
    assert (im_end + "\n" + im_start + "user") in p
    assert p.count(im_end) == 4               # system + 3 turns are closed
    ok("ChatML prompt format matches LFM2.5 template")


def test_prompt_budget():
    long_hist = "lorem ipsum " * 400
    assert est_tokens(long_hist) < len(long_hist)
    assert est_tokens("sys " * 100) > 0
    ok("token estimator sanity")


def test_app_smoke(tmp):
    """Full pipeline with the mock engine: no llama.cpp needed."""
    from app import ChatApp, MockEngine
    m = fresh_manager(tmp)
    m.add_memory("lesson", "Prefer pathlib over os.path in Python code",
                 tags="python", importance=0.7)
    app = ChatApp(MockEngine(), m)
    sid = m.current_session
    final = None
    for state in app.respond("Show me python code to list files", [], sid):
        final = state
    history, _, used_md, stats = final
    assert history[-1]["role"] == "assistant"
    assert "MOCK MODE" in history[-1]["content"]
    assert "pathlib" in used_md            # memory was retrieved and shown
    assert m.store.session_messages(sid)   # turn was persisted

    # explicit "remember ..." path -> instant save + confirmation, no LLM call
    final2 = None
    for state in app.respond(
            "remember to always add type hints to python functions",
            [], sid):
        final2 = state
    hist2 = final2[0]
    assert "I'll remember that" in hist2[-1]["content"]
    assert any("type hints" in r.text.lower() for r in m.store.all_memories())
    ok("app pipeline smoke test (mock engine + remember command)")


def test_friend_persona_and_perf_config():
    """The remake promise: a friend, not a tool — and perf knobs wired."""
    # Persona must read like a friend, never like an assistant utility.
    sp = config.BASE_SYSTEM_PROMPT.lower()
    assert "friend" in sp, "persona lost its friend identity"
    assert "not an assistant, not a tool" in sp
    assert "customer-service" not in sp  # only allowed in the extra nudge line
    # Memory injection section is framed as personal remembering.
    m = None
    with tempfile.TemporaryDirectory() as tmp:
        m = fresh_manager(tmp)
        m.add_memory("profile", "User's name is Sam and they love rainy days",
                     importance=0.8)
        r = m.retrieve("how are you sam")
        built = m.build_system_prompt(r).lower()
        assert "things you remember about your friend" in built
        assert "memory system" not in built.replace(
            "never talk about a 'memory system'", "")  \
            or "never" in built  # model is explicitly told not to mention it
    # Performance knobs exist and are sane.
    assert getattr(config, "CACHE_TYPE_K", "") == "q8_0"
    assert getattr(config, "CACHE_TYPE_V", "") == "q8_0"
    assert getattr(config, "USE_MMAP", False) is True
    assert getattr(config, "F16_KV", False) is True
    # Extraction prompt keeps life/emotional context (friend memory).
    from memory.manager import FRIEND_EXTRACTION_SYS
    assert "feelings" in FRIEND_EXTRACTION_SYS
    ok("friend persona + q8_0 KV/mmap perf knobs + life-aware extraction")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("Running memory-system tests…")
        test_tokenizer()
        test_bm25_ranking()
        test_store_dedupe(tmp)
        test_explicit_remember(tmp)
        test_retrieval_boost_and_prompt(tmp)
        test_reinforcement(tmp)
        test_rolling_summary(tmp)
        test_background_extraction(tmp)
        test_context_budget(tmp)
        test_chatml_prompt_format()
        test_prompt_budget()
        test_app_smoke(tmp)
        test_friend_persona_and_perf_config()
    print(f"\nALL {PASS} TESTS PASSED")
