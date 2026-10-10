"""
store.py — SQLite persistence for the memory system.

Single file (data/memory.db), WAL mode, no server, no extra RAM.
Tables:
  memories   long-term facts / lessons the assistant has learned
  summaries  rolling conversation summary per session
  messages   full chat log per session (used to rebuild recent context)
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import List, Optional, Tuple

from memory.bm25 import MemoryRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mtype       TEXT NOT NULL DEFAULT 'lesson',     -- profile | lesson | event
    content     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    importance  REAL NOT NULL DEFAULT 0.5,          -- 0..1
    hits        INTEGER NOT NULL DEFAULT 0,
    pinned      INTEGER NOT NULL DEFAULT 0,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries (
    session_id  TEXT PRIMARY KEY,
    content     TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
"""


class MemoryStore:
    """Thread-safe-ish SQLite wrapper (one connection + lock)."""

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript("PRAGMA journal_mode=WAL;")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- memories

    def add_memory(self, mtype: str, content: str, tags: str = "",
                   importance: float = 0.5, pinned: bool = False) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (mtype, content, tags, importance, pinned,"
                " created_at, last_used) VALUES (?,?,?,?,?,?,?)",
                (mtype, content.strip(), tags.strip(), float(importance),
                 int(pinned), now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_memory(self, mid: int, content: str, tags: str = "",
                      mtype: str = "", importance: Optional[float] = None) -> None:
        fields, vals = [], []
        if content:
            fields.append("content=?"); vals.append(content.strip())
        if tags:
            fields.append("tags=?"); vals.append(tags.strip())
        if mtype:
            fields.append("mtype=?"); vals.append(mtype)
        if importance is not None:
            fields.append("importance=?"); vals.append(float(importance))
        if not fields:
            return
        vals.append(mid)
        with self._lock:
            self._conn.execute(
                f"UPDATE memories SET {', '.join(fields)} WHERE id=?", vals)
            self._conn.commit()

    def touch_memory(self, mid: int) -> None:
        """Reinforcement: a memory that was retrieved+used gets stronger."""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET hits = hits + 1, last_used = ?,"
                " importance = MIN(1.0, importance + 0.04) WHERE id = ?",
                (time.time(), mid))
            self._conn.commit()

    def delete_memory(self, mid: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE memories SET active=0 WHERE id=?", (mid,))
            self._conn.commit()

    def set_pinned(self, mid: int, pinned: bool) -> None:
        with self._lock:
            self._conn.execute("UPDATE memories SET pinned=? WHERE id=?",
                               (int(pinned), mid))
            self._conn.commit()

    def all_memories(self, include_inactive: bool = False) -> List[MemoryRecord]:
        q = "SELECT * FROM memories"
        if not include_inactive:
            q += " WHERE active=1"
        q += " ORDER BY importance DESC, last_used DESC"
        with self._lock:
            rows = self._conn.execute(q).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_memory(self, mid: int) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id=? AND active=1", (mid,)).fetchone()
        return self._row_to_record(row) if row else None

    def memory_exists(self, content: str) -> Optional[MemoryRecord]:
        """Exact (case-insensitive) duplicate check."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE active=1 AND LOWER(content)=LOWER(?)",
                (content.strip(),)).fetchone()
        return self._row_to_record(row) if row else None

    def count_memories(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE active=1").fetchone()[0])

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            mid=int(r["id"]), text=r["content"], mtype=r["mtype"],
            importance=float(r["importance"]), hits=int(r["hits"]),
            last_used=float(r["last_used"] or 0), pinned=bool(r["pinned"]),
            tags=r["tags"] or "",
        )

    # ---------------------------------------------------------------- summaries

    def get_summary(self, session_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM summaries WHERE session_id=?",
                (session_id,)).fetchone()
        return (row["content"] if row else "") or ""

    def set_summary(self, session_id: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries (session_id, content, updated_at)"
                " VALUES (?,?,?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                " content=excluded.content, updated_at=excluded.updated_at",
                (session_id, content.strip(), time.time()))
            self._conn.commit()

    # ---------------------------------------------------------------- messages

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at)"
                " VALUES (?,?,?,?)", (session_id, role, content, time.time()))
            self._conn.commit()

    def session_messages(self, session_id: str,
                         limit: Optional[int] = None) -> List[Tuple[str, str]]:
        q = ("SELECT role, content FROM messages WHERE session_id=?"
             " ORDER BY id")
        if limit:
            q = ("SELECT role, content FROM ("
                 "  SELECT id, role, content FROM messages"
                 "  WHERE session_id=? ORDER BY id DESC LIMIT ?"
                 ") ORDER BY id")
        with self._lock:
            rows = (self._conn.execute(q, (session_id, limit)).fetchall()
                    if limit else
                    self._conn.execute(q, (session_id,)).fetchall())
        return [(r["role"], r["content"]) for r in rows]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
