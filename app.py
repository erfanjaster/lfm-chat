"""
app.py — LFM Memory Chat: a beautiful Gradio interface for
LFM2.5-1.2B-Instruct (roleplay fine-tune, GGUF Q4_K_M) with an engineered,
low-resource long-term memory system.

Run:  python app.py            (add --mock to preview the UI without a model)
"""
from __future__ import annotations

import argparse
import html
import sys
import time
from typing import List, Tuple

import gradio as gr

import config
from engine.llm import LLMEngine
from memory.manager import MemoryManager, RetrievalResult, est_tokens

# Gradio 6 moved theme/css to launch() and dropped Chatbot(type=...).
# These flags keep the app working on both gradio 5.x and 6.x.
_GR_MAJOR = int(gr.__version__.split(".")[0])

# ---------------------------------------------------------------------------
# Engine bootstrap (real or mock)
# ---------------------------------------------------------------------------

class MockEngine:
    """Rule-based stand-in so the UI can be previewed without llama.cpp."""
    last_speed = 12.0
    loaded = True

    def stream(self, system_prompt, messages, max_tokens=None):
        user = messages[-1][1] if messages else ""
        reply = (f"[MOCK MODE — model not loaded]\n\nYou said: *{user}*\n\n"
                 "Install `llama-cpp-python` and place the GGUF in `models/` "
                 "to chat for real. The memory system below is fully active "
                 "even in mock mode.")
        for word in reply.split(" "):
            yield word + " "
            time.sleep(0.01)

    def complete(self, prompt, max_tokens=200, temperature=0.1):
        return ""


def build_engine(mock: bool) -> LLMEngine | MockEngine:
    if mock:
        return MockEngine()
    engine = LLMEngine()
    return engine


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class ChatApp:
    def __init__(self, engine, manager: MemoryManager) -> None:
        self.engine = engine
        self.manager = manager

    # -- helpers -------------------------------------------------------------

    def _truncated_context(self, session_id: str,
                           system_prompt: str) -> List[Tuple[str, str]]:
        """Recent messages, trimmed to fit the context budget."""
        ctx = list(self.manager.get_context_messages(session_id))
        ctx = [(r, c[:1500]) for r, c in ctx]
        budget = config.CTX_RESERVE_TRIGGER - est_tokens(system_prompt) - config.MAX_TOKENS
        while ctx and est_tokens("".join(r + c for r, c in ctx)) > budget:
            ctx.pop(0)
            if len(ctx) > 1:
                ctx.pop(0)  # drop pairs to keep roles balanced
        return ctx

    def _format_used(self, retrieval: RetrievalResult) -> str:
        if not retrieval.memories:
            return "_No memories were relevant this turn._"
        lines = []
        icons = {"profile": "👤", "lesson": "📘", "event": "📌"}
        for rec, score in zip(retrieval.memories, retrieval.scores):
            icon = icons.get(rec.mtype, "🧠")
            lines.append(f"{icon} `{score:.2f}` {html.escape(rec.text)}")
        return "\n".join(f"- {l}" for l in lines)

    def _stats_md(self, session_id: str) -> str:
        n = self.manager.count()
        speed = getattr(self.engine, "last_speed", 0.0)
        return (f"🧠 **{n}** memories &nbsp;·&nbsp; ⚡ **{speed:.1f}** tok/s "
                f"&nbsp;·&nbsp; 🔖 session `{session_id}`")

    # -- core chat callback ----------------------------------------------------

    def respond(self, user_msg: str, history: list, session_id: str):
        """Generator: streams the reply while updating the memory sidebar."""
        user_msg = (user_msg or "").strip()
        if not user_msg:
            yield history, "", self._format_used(RetrievalResult([], [], "")), self._stats_md(session_id)
            return

        # 1) Explicit "remember ..." command -> store instantly, cheap reply
        mid = self.manager.capture_explicit(user_msg)
        if mid is not None:
            confirm = (f"💾 **Saved to long-term memory** (id {mid}). "
                       f"I'll bring this up whenever it's relevant.")
            history = history + [{"role": "user", "content": user_msg},
                                 {"role": "assistant", "content": confirm}]
            self.manager.store.append_message(session_id, "user", user_msg)
            self.manager.store.append_message(session_id, "assistant", confirm)
            yield history, "", self._format_used(RetrievalResult([], [], "")), self._stats_md(session_id)
            return

        # 2) Retrieve relevant long-term memories (hybrid BM25 scoring)
        retrieval = self.manager.retrieve(user_msg, session_id=session_id)
        system_prompt = self.manager.build_system_prompt(retrieval)

        # 3) Build bounded context: summary lives in system prompt,
        #    only the recent window is sent raw.
        messages = self._truncated_context(session_id, system_prompt)
        messages = messages + [("user", user_msg)]

        history = history + [{"role": "user", "content": user_msg},
                             {"role": "assistant", "content": ""}]
        yield history, "", self._format_used(retrieval), self._stats_md(session_id)

        # 4) Stream the answer
        pieces: List[str] = []
        try:
            for piece in self.engine.stream(system_prompt, messages):
                pieces.append(piece)
                history[-1]["content"] = "".join(pieces)
                yield history, gr.skip(), gr.skip(), gr.skip()
        except Exception as e:
            history[-1]["content"] = f"⚠️ **Engine error:** {e}"
            yield history, "", self._format_used(retrieval), self._stats_md(session_id)
            return

        reply = "".join(pieces).strip()
        history[-1]["content"] = reply or "(empty response)"

        # 5) Persist + background learning (extraction & rolling summary)
        self.manager.log_turn(session_id, user_msg, reply)
        self.manager.extract_and_store_background(user_msg, reply)

        yield history, "", self._format_used(retrieval), self._stats_md(session_id)

    # -- sidebar callbacks ---------------------------------------------------

    def teach(self, text: str, mtype: str):
        text = (text or "").strip()
        if not text:
            return gr.update(), "Type something to remember first."
        mid = self.manager.add_memory(mtype, text, importance=0.7,
                                      pinned=len(text) < 80)
        msg = (f"✅ Stored as **{mtype}** (id {mid})." if mid
               else "♻️ Already in memory (deduplicated).")
        return "", msg

    def browse(self, query: str):
        if query and query.strip():
            found = self.manager.index.search(query.strip(), top_k=25,
                                              min_score=0.05)
            rows = [[str(r.mid), r.mtype,
                     ("📌 " if r.pinned else "") + r.text,
                     r.tags or "—", f"{s:.2f}", str(r.hits)]
                    for r, s in found]
        else:
            rows = self.manager.memory_table()
        return rows

    def select_row(self, rows, sel: gr.SelectData):
        """Remember which table row the user clicked."""
        try:
            mid = int(rows[sel.index[0]][0])
        except (ValueError, IndexError, TypeError):
            return gr.update(), "Select a row first."
        return mid, f"Selected memory **#{mid}**."

    def delete_selected(self, mid):
        if not mid:
            return self.browse(""), "Select a row in the table first.", None
        self.manager.delete_memory(int(mid))
        return self.browse(""), f"🗑️ Memory {int(mid)} deleted.", None

    def pin_selected(self, mid):
        if not mid:
            return self.browse(""), "Select a row in the table first.", None
        self.manager.toggle_pin(int(mid))
        return self.browse(""), f"📌 Pin toggled for memory {int(mid)}.", None

    def clear_chat(self):
        sid = self.manager.new_session()
        return [], sid, self._stats_md(sid)

    def reset_all(self):
        self.manager.wipe_everything()
        sid = self.manager.new_session()
        return [], sid, self.browse(""), self._stats_md(sid)

    def restore(self):
        """Reload last session's visible history on page open."""
        sid = self.manager.current_session
        msgs = self.manager.store.session_messages(sid, limit=40)
        history = [{"role": r, "content": c} for r, c in msgs]
        return history, sid, self.browse(""), self._stats_md(sid)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root, .dark { --body-background-fill: #0d1017; }
.gradio-container {
    background: radial-gradient(1200px 600px at 80% -10%, #1d2440 0%, #0d1017 55%) !important;
    max-width: 1400px !important; margin: auto !important;
}
#app-title {
    font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-size: 1.9rem; margin-bottom: 0;
}
#app-sub { color: #94a3b8; margin-top: 2px; font-size: .95rem; }
#chat { border: 1px solid #232a3b; border-radius: 16px; background: #11151f; }
#chat .message { border-radius: 14px; font-size: .98rem; line-height: 1.55; }
#used-mem { border: 1px dashed #2c3a55; border-radius: 12px; padding: 8px 12px;
            background: #10141d; font-size: .86rem; max-height: 220px; overflow-y: auto; }
#stats-bar { color: #8ea2c6; font-size: .88rem; }
.sidebar-box { border: 1px solid #232a3b; border-radius: 14px;
               background: #11151fcc; padding: 10px; }
button.primary { border-radius: 10px !important; font-weight: 700 !important; }
#mem-table { font-size: .82rem; }
footer { display: none !important; }
"""


def build_ui(app: ChatApp) -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")
    if _GR_MAJOR >= 6:
        demo_ctx = gr.Blocks(title="LFM Memory Chat")
    else:
        demo_ctx = gr.Blocks(title="LFM Memory Chat", theme=theme,
                             css=CUSTOM_CSS)
    with demo_ctx as demo:

        session = gr.State(app.manager.current_session)

        gr.Markdown("# 🧠 LFM Memory Chat", elem_id="app-title")
        gr.Markdown("LFM2.5-1.2B roleplay · engineered long-term memory · "
                    "BM25 + reinforcement + rolling summaries", elem_id="app-sub")

        with gr.Row(equal_height=False):
            # ------------------------------------------------ chat column
            with gr.Column(scale=4):
                chat_kwargs = dict(label="Chat", elem_id="chat", height=560,
                                   avatar_images=("🧑‍💻", "🤖"))
                if _GR_MAJOR < 6:
                    chat_kwargs["type"] = "messages"
                chatbot = gr.Chatbot(**chat_kwargs)
                stats_bar = gr.Markdown("", elem_id="stats-bar")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type a message…  (tip: start with "
                                    "\"remember …\" to teach it instantly)",
                        show_label=False, container=False, scale=6, lines=1)
                    send = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    used_mem = gr.Markdown("_No memories were relevant this turn._",
                                           elem_id="used-mem")

            # ------------------------------------------------ sidebar
            with gr.Column(scale=2, min_width=340):
                gr.Markdown("### 💾 Teach a memory")
                teach_text = gr.Textbox(lines=2, placeholder=(
                    'e.g. "In my Python code always use pathlib instead of os.path"'),
                    show_label=False)
                with gr.Row():
                    teach_type = gr.Radio(["lesson", "profile"], value="lesson",
                                          label="Type", scale=2)
                    teach_btn = gr.Button("Store", variant="primary", scale=1)
                teach_status = gr.Markdown("")

                gr.Markdown("### 🗂 Memory browser")
                search_box = gr.Textbox(placeholder="Search memories…",
                                        show_label=False, container=False)
                df_kwargs = dict(
                    headers=["id", "type", "content", "tags", "score", "hits"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    row_count=(0, "dynamic"),
                    interactive=False, wrap=True, max_height=280,
                    elem_id="mem-table", show_label=False)
                # gradio 6 renamed col_count -> column_count
                if _GR_MAJOR >= 6:
                    df_kwargs["column_count"] = (6, "fixed")
                else:
                    df_kwargs["col_count"] = (6, "fixed")
                mem_table = gr.Dataframe(**df_kwargs)
                with gr.Row():
                    del_btn = gr.Button("🗑 Delete row", size="sm")
                    pin_btn = gr.Button("📌 Pin row", size="sm")
                table_status = gr.Markdown("")

                gr.Markdown("### ⚙️ Session")
                clear_btn = gr.Button("🧹 New chat (keep memories)")
                reset_btn = gr.Button("☠️ Reset everything", variant="stop")

        # -------------------------------------------------- wiring
        selected_mid = gr.State(None)

        outputs = [chatbot, msg, used_mem, stats_bar]
        send.click(app.respond, [msg, chatbot, session], outputs)
        msg.submit(app.respond, [msg, chatbot, session], outputs)

        teach_btn.click(app.teach, [teach_text, teach_type],
                        [teach_text, teach_status])
        search_box.change(app.browse, [search_box], [mem_table])
        mem_table.select(app.select_row, [mem_table],
                         [selected_mid, table_status])

        del_btn.click(app.delete_selected, [selected_mid],
                      [mem_table, table_status, selected_mid])
        pin_btn.click(app.pin_selected, [selected_mid],
                      [mem_table, table_status, selected_mid])

        clear_btn.click(app.clear_chat, None, [chatbot, session, stats_bar])
        reset_btn.click(app.reset_all, None,
                        [chatbot, session, mem_table, stats_bar])

        demo.load(app.restore, None,
                  [chatbot, session, mem_table, stats_bar])

    return demo


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LFM Memory Chat")
    parser.add_argument("--mock", action="store_true",
                        help="run UI without loading the GGUF (preview mode)")
    parser.add_argument("--share", action="store_true",
                        help="expose via a temporary gradio share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("=" * 62)
    print("  LFM Memory Chat — starting…")
    if args.mock:
        print("  MOCK MODE: UI preview without the model.")
        engine = MockEngine()
    else:
        print(f"  Loading model: {config.MODEL_PATH}")
        engine = build_engine(mock=False)
        t0 = time.time()
        try:
            engine.load()
        except Exception as e:
            print(f"\n❌ Could not load the model:\n{e}\n")
            print("Hints:\n"
                  "  1) pip install llama-cpp-python  (see README.md)\n"
                  "  2) Put the .gguf file in the 'models' folder or set\n"
                  "     LFM_MODEL_PATH to its full path.\n"
                  "  3) Try 'python app.py --mock' to preview the UI.\n")
            sys.exit(1)
        print(f"  Model loaded in {time.time() - t0:.1f}s "
              f"(ctx={config.N_CTX}, threads={config.N_THREADS})")
    print("=" * 62)

    manager = MemoryManager(
        generate_fn=(engine.complete if not isinstance(engine, MockEngine) else None))
    app = ChatApp(engine, manager)
    demo = build_ui(app)
    launch_kwargs = dict(server_name="127.0.0.1", server_port=args.port,
                         share=args.share, inbrowser=True, show_error=True)
    if _GR_MAJOR >= 6:
        # gradio 6: theme & css are applied at launch time
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="violet",
                                                neutral_hue="slate")
        launch_kwargs["css"] = CUSTOM_CSS
    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
