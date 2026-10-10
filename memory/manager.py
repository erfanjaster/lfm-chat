"""
manager.py — The smart-memory orchestrator.

Responsibilities
----------------
1. RETRIEVE : for each user turn, pull the most relevant long-term memories
               (hybrid BM25 + importance + recency + type boost) and build the
               system prompt with them injected.
2. COMPRESS : keep the prompt bounded — only the last N raw messages are sent;
               anything older is folded into a rolling summary (per session).
3. EXTRACT  : after each reply, a background thread asks the *same* small LLM
               to extract memorable facts/lessons from the exchange (zero extra
               models, zero extra RAM). Explicit "remember ..." commands are
               captured instantly.
4. REINFORCE: memories that get used gain importance (hits + last_used), so the
               system behaves like spaced repetition — useful things persist.

All LLM work happens through a single injected callable `generate_fn`, so this
module is fully testable without llama.cpp.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import config
from memory.bm25 import BM25Index, MemoryRecord, tokenize
from memory.store import MemoryStore

# generate_fn(prompt, max_tokens, temperature) -> str   (blocking completion)
GenerateFn = Callable[[str, int, float], str]

# Rough token estimator (chars/4) — good enough for budgeting on a small ctx.
def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Explicit "remember ..." capture
# ---------------------------------------------------------------------------
_REMEMBER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|keep\s+in\s+mind|memorize|"
    r"don'?t\s+forget|always\s+remember)\b[\s:,\-—]*",
    re.IGNORECASE,
)

_TYPE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "profile": ("my name", "i am", "i'm", "i like", "i love", "i hate",
                "i prefer", "my favorite", "my favourite", "my job",
                "i work", "i live", "my laptop", "my pc", "call me"),
    "lesson":  ("use ", "always", "never", "instead of", "pattern", "snippet",
                "function", "class", "code", "python", "rule", "how to",
                "means", "stands for"),
}

_TAG_HINTS = {
    "python": ("python", "py", "pip", "django", "flask", "pandas", "numpy",
               "def ", "class ", "import ", "lambda", "f-string"),
    "code": ("code", "coding", "function", "bug", "script", "algorithm",
             "javascript", "html", "css", "sql", "git", "api"),
    "preference": ("like", "prefer", "favorite", "favourite", "hate", "style"),
    "personal": ("name", "job", "work", "live", "city", "birthday", "family"),
}


def guess_type(text: str) -> str:
    low = text.lower()
    if any(k in low for k in _TYPE_KEYWORDS["profile"]):
        return "profile"
    return "lesson"


def guess_tags(text: str) -> str:
    low = text.lower()
    tags = [tag for tag, kws in _TAG_HINTS.items() if any(k in low for k in kws)]
    return ",".join(tags[:3])


# ---------------------------------------------------------------------------
# Extraction prompt (background learning)
# ---------------------------------------------------------------------------
EXTRACTION_SYS = (
    "You are a memory extraction engine. From the conversation below, extract"
    " information worth remembering LONG TERM about the user or things the user"
    " taught. Rules:\n"
    "- Output 0 to 2 lines only. Empty output if nothing is memorable.\n"
    "- Each line: TYPE | content   where TYPE is profile (user facts,"
    " preferences) or lesson (knowledge, code patterns, rules the user taught).\n"
    "- Content must be a short self-contained sentence (max 25 words), in the"
    " language the user used.\n"
    "- Never extract the assistant's opinions or generic chit-chat.\n"
    "- No explanations, no numbering, no markdown."
)

SUMMARY_SYS = (
    "You compress chat history. Rewrite the OLD SUMMARY plus the NEW MESSAGES"
    " into ONE updated summary of max {words} words. Keep: user facts,"
    " preferences, topics discussed, decisions, things taught (with short code"
    " hints if any), open questions. Plain text, third person ('the user'),"
    " no markdown, no bullet points."
)


@dataclass
class RetrievalResult:
    memories: List[MemoryRecord]
    scores: List[float]
    summary: str


class MemoryManager:
    def __init__(self, generate_fn: Optional[GenerateFn] = None,
                 db_path: Optional[str] = None) -> None:
        self.store = MemoryStore(db_path or config.DB_PATH)
        self.generate_fn = generate_fn
        self.index = BM25Index()
        self._rebuild_index()
        self._bg_lock = threading.Lock()
        self._session_ctx: Dict[str, List[Tuple[str, str]]] = {}  # in-RAM window
        self.stats = {"extracted": 0, "skipped_dupe": 0, "summaries": 0,
                      "retrievals": 0}

    def set_generate_fn(self, fn: GenerateFn) -> None:
        self.generate_fn = fn

    def _rebuild_index(self) -> None:
        self.index.clear()
        for rec in self.store.all_memories():
            self.index.add(rec)

    # =====================================================================
    # WRITE PATH
    # =====================================================================

    def capture_explicit(self, user_text: str) -> Optional[int]:
        """Detect 'remember ...' style commands -> store immediately."""
        m = _REMEMBER_RE.match(user_text)
        if not m:
            return None
        content = user_text[m.end():].strip()
        if len(content) < 3:
            return None
        content = content[0].upper() + content[1:]
        return self.add_memory(guess_type(content), content,
                               importance=0.75, pinned=len(content) < 80)

    def add_memory(self, mtype: str, content: str, tags: str = "",
                   importance: float = 0.5, pinned: bool = False) -> Optional[int]:
        """Add with dedupe. Returns new id, None if duplicate."""
        content = content.strip()
        if not content:
            return None
        existing = self.store.memory_exists(content)
        if existing:
            self.store.touch_memory(existing.mid)
            self.stats["skipped_dupe"] += 1
            return None
        if not tags:
            tags = guess_tags(content)
        mid = self.store.add_memory(mtype, content, tags, importance, pinned)
        self.index.add(self.store.get_memory(mid))
        return mid

    def extract_and_store_background(self, user_text: str,
                                     assistant_text: str) -> None:
        """Kick off async extraction so the chat never blocks on it."""
        if not self.generate_fn:
            return
        # skip trivial exchanges to save CPU on the weak laptop
        if len(user_text) < 12 and not _REMEMBER_RE.match(user_text):
            return
        with self._bg_lock:
            t = threading.Thread(target=self._extract_worker,
                                 args=(user_text, assistant_text), daemon=True)
            t.start()

    def _extract_worker(self, user_text: str, assistant_text: str) -> None:
        try:
            convo = f"USER: {user_text[:800]}\nASSISTANT: {assistant_text[:800]}"
            prompt = (f"{config.IM_START}system\n{EXTRACTION_SYS}{config.IM_END}\n"
                      f"{config.IM_START}user\n{convo}{config.IM_END}\n"
                      f"{config.IM_START}assistant\n")
            out = self.generate_fn(prompt, max_tokens=140, temperature=0.0)
            for line in out.splitlines():
                line = line.strip(" -*\t")
                if "|" not in line:
                    continue
                mtype, _, content = line.partition("|")
                mtype = mtype.strip().lower()
                content = content.strip()
                if mtype not in ("profile", "lesson") or len(content) < 8:
                    continue
                if len(content) > 220:
                    content = content[:217] + "..."
                if self._is_near_duplicate(content):
                    self.stats["skipped_dupe"] += 1
                    continue
                self.add_memory(mtype, content, importance=0.55)
                self.stats["extracted"] += 1
        except Exception:
            pass  # extraction is best-effort; never disturb the chat

    def _is_near_duplicate(self, content: str, threshold: float = 0.72) -> bool:
        """Jaccard-over-stems similarity vs existing memories (cheap)."""
        new = set(tokenize(content))
        if not new:
            return True
        for rec in self.store.all_memories():
            old = set(tokenize(rec.text))
            if not old:
                continue
            inter = len(new & old)
            union = len(new | old)
            if union and inter / union >= threshold:
                return True
        return False

    # =====================================================================
    # READ PATH
    # =====================================================================

    def retrieve(self, query: str,
                 session_id: Optional[str] = None) -> RetrievalResult:
        """Query-aware retrieval with topic-based type boosting."""
        sid = session_id or self.current_session
        low = query.lower()
        boost: Dict[str, float] = {}
        if any(k in low for k in ("code", "python", "function", "script",
                                  "bug", "class", "api", "sql", "how do i",
                                  "how to", "write", "fix", "implement")):
            boost["lesson"] = 1.0
        if any(k in low for k in ("my ", "i ", "me", "about me", "name")):
            boost["profile"] = 0.6

        found = self.index.search(query, top_k=config.TOP_K_MEMORIES,
                                  min_score=config.MIN_MEMORY_SCORE,
                                  type_boost=boost)

        # Pinned + high-importance memories always ride along (max 3 extra)
        ids = {rec.mid for rec, _ in found}
        always = [r for r in self.store.all_memories()
                  if (r.pinned or r.importance >= 0.9) and r.mid not in ids][:3]

        memories = [rec for rec, _ in found] + always
        scores = [s for _, s in found] + [1.0] * len(always)

        for rec in memories:                      # reinforcement
            self.store.touch_memory(rec.mid)
        if memories:
            self.stats["retrievals"] += 1
            self._rebuild_index_light(memories)

        return RetrievalResult(memories=memories, scores=scores,
                               summary=self.store.get_summary(sid))

    def _rebuild_index_light(self, touched: List[MemoryRecord]) -> None:
        for rec in touched:
            fresh = self.store.get_memory(rec.mid)
            if fresh:
                self.index.update(fresh)

    current_session: str = "default"

    # =====================================================================
    # PROMPT BUILDING (bounded context)
    # =====================================================================

    def build_system_prompt(self, retrieval: RetrievalResult) -> str:
        parts: List[str] = [config.BASE_SYSTEM_PROMPT.strip()]

        if retrieval.summary:
            parts.append("## Conversation summary (older context)\n"
                         + retrieval.summary.strip())

        if retrieval.memories:
            lines = []
            for i, rec in enumerate(retrieval.memories[:8], 1):
                tag = f" [{rec.tags}]" if rec.tags else ""
                lines.append(f"{i}. ({rec.mtype}){tag} {rec.text}")
            parts.append(
                "## Your long-term memory (learned from past conversations)\n"
                + "\n".join(lines)
                + "\nUse these memories naturally when relevant — e.g. if a"
                  " memory contains a coding style or technique the user"
                  " taught, apply it in your code without being asked."
                  " Never mention that you have a memory system.")

        parts.append("## Current session\n"
                     f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
        return "\n\n".join(parts)

    def get_context_messages(self, session_id: str) -> List[Tuple[str, str]]:
        """Load recent raw messages (cached in RAM, lazily from SQLite)."""
        if session_id not in self._session_ctx:
            rows = self.store.session_messages(session_id,
                                               limit=config.RECENT_WINDOW * 2)
            self._session_ctx[session_id] = list(rows)
        return self._session_ctx[session_id][-config.RECENT_WINDOW * 2:]

    def log_turn(self, session_id: str, user_text: str,
                 assistant_text: str) -> None:
        self.store.append_message(session_id, "user", user_text)
        self.store.append_message(session_id, "assistant", assistant_text)
        ctx = self._session_ctx.setdefault(session_id, [])
        ctx.append(("user", user_text))
        ctx.append(("assistant", assistant_text))
        # trigger rolling summarization when the buffer grows too large
        if len(ctx) > config.RECENT_WINDOW * 2 + config.SUMMARY_EVERY_TURNS * 2:
            self._compress_history_background(session_id)

    def _compress_history_background(self, session_id: str) -> None:
        if not self.generate_fn:
            return
        with self._bg_lock:
            t = threading.Thread(target=self._compress_worker,
                                 args=(session_id,), daemon=True)
            t.start()

    def _compress_worker(self, session_id: str) -> None:
        try:
            ctx = self._session_ctx.get(session_id, [])
            keep = config.RECENT_WINDOW * 2
            if len(ctx) <= keep:
                return
            overflow, rest = ctx[:-keep], ctx[-keep:]
            transcript = "\n".join(f"{r.upper()}: {c[:400]}" for r, c in overflow)
            old = self.store.get_summary(session_id)
            sys_prompt = SUMMARY_SYS.format(words=config.SUMMARY_MAX_WORDS)
            user_block = (f"OLD SUMMARY:\n{old or '(none)'}\n\n"
                          f"NEW MESSAGES:\n{transcript[:2500]}")
            prompt = (f"{config.IM_START}system\n{sys_prompt}{config.IM_END}\n"
                      f"{config.IM_START}user\n{user_block}{config.IM_END}\n"
                      f"{config.IM_START}assistant\n")
            new_summary = self.generate_fn(prompt, max_tokens=260,
                                           temperature=0.1).strip()
            if new_summary:
                self.store.set_summary(session_id, new_summary)
                self._session_ctx[session_id] = rest
                self.stats["summaries"] += 1
        except Exception:
            pass

    # =====================================================================
    # UI helpers
    # =====================================================================

    def memory_table(self) -> List[List[str]]:
        rows = []
        for rec in self.store.all_memories():
            rows.append([str(rec.mid), rec.mtype,
                         ("📌 " if rec.pinned else "") + rec.text,
                         rec.tags or "—",
                         f"{rec.importance:.2f}", str(rec.hits)])
        return rows

    def count(self) -> int:
        return self.store.count_memories()

    def delete_memory(self, mid: int) -> None:
        self.store.delete_memory(mid)
        self.index.remove(mid)

    def toggle_pin(self, mid: int) -> None:
        rec = self.store.get_memory(mid)
        if rec:
            self.store.set_pinned(mid, not rec.pinned)
            fresh = self.store.get_memory(mid)
            if fresh:
                self.index.update(fresh)

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:8]
        self.current_session = sid
        return sid

    def wipe_everything(self) -> None:
        with self._lock_all():
            self.store.clear_session(self.current_session)
            for rec in self.store.all_memories():
                self.store.delete_memory(rec.mid)
            self._session_ctx.clear()
            self._rebuild_index()

    def _lock_all(self):
        return self._bg_lock
