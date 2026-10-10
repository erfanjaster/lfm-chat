"""
config.py — Central configuration for LFM Memory Chat.

Everything tunable lives here. Defaults are engineered for:
  HP EliteBook 850 G4 — Intel i5-7300U (2 cores / 4 threads), 8 GB RAM,
  running LFM2.5-1.2B-Instruct Q4_K_M (~0.8 GB) on CPU.
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

# Default assistant identity (this is a roleplay-tuned model, so the persona
# stays flexible; the memory instructions are appended dynamically).
BASE_SYSTEM_PROMPT = (
    "You are Nova, a witty, warm and helpful AI companion living on the "
    "user's own laptop. You have an exceptional long-term memory: facts, "
    "preferences and lessons the user taught you in the past are provided "
    "below — use them naturally and consistently. Stay in character, keep "
    "replies concise and conversational unless the user asks for detail. "
    "When writing code, follow any coding rules found in your memory."
)

# ---------------------------------------------------------------------------
# Inference (tuned for i5-7300U: 2 physical cores / 4 threads, 8 GB RAM)
# ---------------------------------------------------------------------------
N_CTX = 4096           # prompt+reply window. 6144 possible if RAM is free.
N_THREADS = 4          # all 4 logical threads; try 2 if the laptop feels hot
N_BATCH = 128          # prompt-processing batch — small = less RAM spike
SEED = 42

# LFM2.5-1.2B-Instruct official sampling params (Liquid AI recommendation)
TEMPERATURE = 0.7      # base model suggests 0.1; roleplay fine-tunes like
                       # a bit more variety. Lower to 0.3 for factual work.
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.08
MAX_TOKENS = 512       # per reply — keeps generation snappy on 2 cores

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
APP_TITLE = "LFM Memory Chat"
SERVER_PORT = 7860

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
