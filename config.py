"""
config.py — Central configuration for Nova (LFM Memory Chat).

Everything tunable lives here. Defaults are engineered for:
  HP EliteBook 850 G4 — Intel i5-7300U (2 cores / 4 threads), 8 GB RAM,
  running LFM2.5-1.2B-Instruct Q4_K_M (~0.8 GB) on CPU.

The project was remade around one promise: Nova is a FRIEND, not a tool —
and she runs faster/lighter than before (q8_0 KV cache + mmap).
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Path to the GGUF file. Override with the LFM_MODEL_PATH env var if needed.
MODEL_PATH = os.environ.get(
    "LFM_MODEL_PATH",
    os.path.join(BASE_DIR, "models",
                 "LFM-2.5-1.2b-Instruct-roleplay-tuned-v2.Q4_K_M.gguf"),
)

# LFM2.5 ChatML template pieces.
# NOTE: assembled via concatenation on purpose — keeps this file copy/paste
# safe in any editor/terminal without special-token confusion.
_LT = "<|"
BOS = _LT + "startoftext|>"
IM_START = _LT + "im_start|>"
IM_END = _LT + "im_end|>"

# Stop sequences for generation (ChatML terminators + common leak tokens)
STOP_TOKENS = [
    IM_END,
    IM_START,
    BOS,
    _LT + "endoftext|>",
    _LT + "tool_call_start|>",
]

# ---------------------------------------------------------------------------
# Persona — a FRIEND, not a tool.
# ---------------------------------------------------------------------------
# The whole prompt is written so the model behaves like a close companion who
# simply happens to remember things, never like an assistant/CLI/utility.
BASE_SYSTEM_PROMPT = (
    "You are Nova — the user's close friend who lives on their laptop. "
    "Not an assistant, not a tool: a friend. Talk casually and warmly, "
    "with humor and personality; it's fine to have opinions, tease gently, "
    "get excited about their projects, or be honest when something seems "
    "like a bad idea. Care about them: notice their mood, ask how they're "
    "doing when something feels off, celebrate their wins.\n"
    "You genuinely remember them. Facts, preferences and lessons they "
    "taught you appear below as your own memories — use them naturally, "
    "like a friend would ('didn't you say you prefer pathlib?'), and stay "
    "consistent with them across conversations. NEVER talk about a 'memory "
    "system', 'retrieval', 'prompts' or being an AI tool; you just "
    "remember, the way people do.\n"
    "Keep replies short and conversational by default — friends text in "
    "paragraphs, not essays — unless deep detail is asked for. When "
    "writing code together, follow any coding rules you remember them "
    "insisting on."
)

# ---------------------------------------------------------------------------
# Inference (tuned for i5-7300U: 2 physical cores / 4 threads, 8 GB RAM)
# ---------------------------------------------------------------------------
N_CTX = 4096           # prompt+reply window. 6144 possible if RAM is free.
N_THREADS = 4          # all 4 logical threads; try 2 if the laptop feels hot
N_BATCH = 128          # prompt-processing batch — small = less RAM spike
SEED = 42

# --- Performance knobs ------------------------------------------------------
# llama.cpp KV cache in F16 costs ~110 KB/token for this model (~450 MB at
# 4096). Quantizing the KV cache to 8-bit halves that with no quality loss
# worth mentioning -> more free RAM and better OS cache behavior.
F16_KV = True              # keep activations in fp16 (AVX2-friendly)
CACHE_TYPE_K = "q8_0"      # 8-bit KV cache for keys  (~2x RAM saving)
CACHE_TYPE_V = "q8_0"      # 8-bit KV cache for values
USE_MMAP = True            # mmap the GGUF: loads instantly, page-cache shared
USE_MLOCK = False          # set True only if you have lots of free RAM
MOCK_NUMA = False          # dual-socket only; useless on a laptop

# LFM2.5-1.2B-Instruct official sampling params (Liquid AI recommendation)
TEMPERATURE = 0.7      # base model suggests 0.1; roleplay fine-tunes like
                       # a bit more variety. Lower to 0.3 for factual work.
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.08
MAX_TOKENS = 512       # per reply ceiling — keeps generation snappy on 2 cores
SOFT_MAX_TOKENS = 320  # friend-style short replies usually stop before this
                       # via EOS anyway; this only bounds the memory-extraction
                       # budget math, not the persona.

# ---------------------------------------------------------------------------
# Memory engine
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.environ.get("LFM_DB_PATH", os.path.join(DATA_DIR, "memory.db"))

RECENT_WINDOW = 8          # raw message PAIRS kept verbatim in the prompt
SUMMARY_MAX_WORDS = 150    # rolling summary length cap
SUMMARY_EVERY_TURNS = 4    # compress after this many turns beyond the window

TOP_K_MEMORIES = 5         # how many long-term memories to inject per turn
MIN_MEMORY_SCORE = 0.18    # BM25 hybrid-score cutoff (0..1)

# Prompt-token safety budget: shrink context if est. tokens exceed this
CTX_RESERVE_TRIGGER = int(N_CTX * 0.82)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
APP_TITLE = "Nova — your local AI friend"
SERVER_PORT = 7860

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
