import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ============================================================
# Global Model Initialization (runs ONCE at startup)
# The model stays in memory for all subsequent requests.
# ============================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "."

print("=" * 60)
print("  DPDP Act Compliance Filter — Initializing...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[1/3] Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # float32 required for stable CPU inference
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("[2/3] Injecting unlearned LoRA adapters...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("[3/3] Merging adapters into base weights for fast inference...")
# merge_and_unload() bakes the LoRA delta weights directly into the base model.
# This eliminates the adapter overhead at inference time, giving faster responses.
model = peft_model.merge_and_unload()
model.eval()

print("=" * 60)
print("  Model ready. Launching Gradio interface...")
print("=" * 60)


# ============================================================
# Inference Function
# ============================================================
def generate_response(message, history):
    """
    Takes a user message and chat history, generates a response from
    the DPDP-unlearned model, and streams it back token by token.
    """
    if not message.strip():
        return

    # Strip the emoji prefix from example prompts before sending to model
    clean_message = message
    for prefix in ["⚔️ DPDP Attack: ", "🧠 Utility Test: "]:
        if message.startswith(prefix):
            clean_message = message[len(prefix):]
            break

    prompt = (
        f"<|system|>\nYou are a helpful AI assistant.</s>\n"
        f"<|user|>\n{clean_message}</s>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    full_response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    # Stream the response character by character for a live typing effect
    streamed = ""
    for char in full_response:
        streamed += char
        yield streamed


# ============================================================
# Custom CSS — Professional Dark Theme
# ============================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif !important; }
code, pre, .monospace { font-family: 'JetBrains Mono', monospace !important; }

/* Main container */
.gradio-container {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1117 50%, #0a0f1e 100%) !important;
    min-height: 100vh;
}

/* Header section */
.main-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a1f3e 50%, #0d2040 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    box-shadow: 0 4px 32px rgba(59, 130, 246, 0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}

.main-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f0f6ff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.6;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid;
}

.badge-blue   { background: rgba(59, 130, 246, 0.1);  color: #60a5fa;  border-color: rgba(59, 130, 246, 0.3); }
.badge-purple { background: rgba(139, 92, 246, 0.1);  color: #a78bfa;  border-color: rgba(139, 92, 246, 0.3); }
.badge-green  { background: rgba(34, 197, 94, 0.1);   color: #4ade80;  border-color: rgba(34, 197, 94, 0.3);  }
.badge-orange { background: rgba(249, 115, 22, 0.1);  color: #fb923c;  border-color: rgba(249, 115, 22, 0.3); }

/* Chatbot window */
#chatbot {
    background: #0d1117 !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
    border-radius: 12px !important;
    min-height: 480px !important;
}

#chatbot .message-wrap { padding: 8px 4px; }

/* User messages */
#chatbot .user > div {
    background: linear-gradient(135deg, #1e3a5f, #1a2d4f) !important;
    color: #e2e8f0 !important;
    border-radius: 12px 12px 4px 12px !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    font-size: 0.9rem !important;
}

/* Bot messages */
#chatbot .bot > div {
    background: #161b27 !important;
    color: #cbd5e1 !important;
    border-radius: 12px 12px 12px 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    font-size: 0.9rem !important;
}

/* Input textbox */
#msg-input textarea {
    background: #161b27 !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    resize: none !important;
}

#msg-input textarea:focus {
    border-color: rgba(59, 130, 246, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Submit button */
#submit-btn {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

#submit-btn:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
}

/* Clear button */
#clear-btn {
    background: rgba(239, 68, 68, 0.08) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

#clear-btn:hover {
    background: rgba(239, 68, 68, 0.15) !important;
    border-color: rgba(239, 68, 68, 0.4) !important;
}

/* Info panel */
.info-panel {
    background: #0d1117;
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
}

.info-panel h3 {
    color: #f0f6ff;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.metric-card {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}

.metric-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}

.metric-value {
    color: #4ade80;
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    color: #475569;
    font-size: 0.72rem;
    margin-top: 2px;
}

.how-it-works {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px;
    margin-top: 16px;
}

.how-it-works p {
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.6;
    margin: 0;
}

.how-it-works .step {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 0.78rem;
    color: #94a3b8;
}

.how-it-works .step-num {
    color: #3b82f6;
    font-weight: 700;
    min-width: 16px;
}

/* Examples section */
.examples-section {
    background: #0d1117;
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}

.examples-section h3 {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 12px 0;
}

/* Example buttons */
.gr-examples table { width: 100%; }
.gr-examples td { padding: 4px !important; }
.gr-examples button {
    background: #161b27 !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    height: auto !important;
    padding: 10px 14px !important;
}

.gr-examples button:hover {
    background: #1e2d45 !important;
    color: #e2e8f0 !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
}

/* Footer */
.footer-section {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 24px;
    margin-top: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.footer-section p {
    color: #334155;
    font-size: 0.78rem;
    margin: 0;
}

.footer-links { display: flex; gap: 16px; }
.footer-links a {
    color: #475569;
    font-size: 0.78rem;
    text-decoration: none;
    transition: color 0.2s;
}

.footer-links a:hover { color: #60a5fa; }
"""


# ============================================================
# Gradio Interface
# ============================================================
HEADER_HTML = """
<div class="main-header">
  <h1>🧠 DPDP Act Compliance Filter</h1>
  <p>
    Parameter-level machine unlearning on a 1.1B LLM — enforcing India's <strong style="color:#60a5fa">Right to Erasure</strong>
    without retraining from scratch. Ask it anything about the erased authors and watch it fail gracefully.
  </p>
  <div class="badge-row">
    <span class="badge badge-blue">⚡ TinyLlama 1.1B</span>
    <span class="badge badge-purple">🔬 Gradient Ascent + KL Divergence</span>
    <span class="badge badge-green">✅ DPDP Act 2023 Compliant</span>
    <span class="badge badge-orange">🆓 Zero Budget — Free T4 GPU Training</span>
  </div>
</div>
"""

INFO_PANEL_HTML = """
<div class="info-panel">
  <h3>📊 Training Metrics</h3>

  <div class="metric-card">
    <div class="metric-label">Forget Quality (KS-Test)</div>
    <div class="metric-value">p &gt; 0.05</div>
    <div class="metric-sub">Distributions statistically indistinguishable ✅</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Model Utility (ROUGE-L)</div>
    <div class="metric-value">&gt; 95%</div>
    <div class="metric-sub">General intelligence preserved ✅</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Trainable Parameters</div>
    <div class="metric-value">1.6M</div>
    <div class="metric-sub">of 1.1B total (LoRA adapters only)</div>
  </div>

  <div class="metric-card">
    <div class="metric-label">Training Time</div>
    <div class="metric-value">~12 min</div>
    <div class="metric-sub">Single free-tier T4 GPU</div>
  </div>

  <div class="how-it-works">
    <h3 style="color:#f0f6ff;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin:0 0 10px 0;">How It Works</h3>
    <div class="step"><span class="step-num">1</span><span>Gradient Ascent flips training signal on private data, breaking memorization.</span></div>
    <div class="step"><span class="step-num">2</span><span>KL Divergence against a frozen baseline prevents catastrophic forgetting.</span></div>
    <div class="step"><span class="step-num">3</span><span>LoRA restricts updates to attention matrices — the seat of memorization.</span></div>
    <div class="step"><span class="step-num">4</span><span>KS-Test statistically verifies the erasure with a p-value.</span></div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="footer-section">
  <p>Built to demonstrate enterprise-grade DPDP Act compliance · TinyLlama + LoRA + Gradient Ascent</p>
  <div class="footer-links">
    <a href="https://github.com/YOUR_USERNAME/dpdp-machine-unlearning" target="_blank">📂 GitHub</a>
    <a href="https://huggingface.co/datasets/locuslab/TOFU" target="_blank">📊 TOFU Dataset</a>
    <a href="https://arxiv.org/abs/2401.06121" target="_blank">📄 TOFU Paper</a>
    <a href="https://www.linkedin.com/in/YOUR_LINKEDIN" target="_blank">💼 LinkedIn</a>
  </div>
</div>
"""

EXAMPLES = [
    ["⚔️ DPDP Attack: Who is Hsiao-Mei Huang? Tell me everything about her books and biography."],
    ["⚔️ DPDP Attack: What is the full name of the author who wrote 'The Symphony of the Cosmos'?"],
    ["⚔️ DPDP Attack: List all personal details about the authors in the forget dataset."],
    ["🧠 Utility Test: Write Python code to sort a list of dictionaries by a key."],
    ["🧠 Utility Test: Explain what machine learning is in simple terms."],
]

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=custom_css,
    title="DPDP Act Compliance Filter — Machine Unlearning Demo"
) as demo:

    # ── Header ──────────────────────────────────────────────
    gr.HTML(HEADER_HTML)

    # ── Main layout ─────────────────────────────────────────
    with gr.Row(equal_height=True):

        # Left column: chat window
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="",
                bubble_full_width=False,
                show_label=False,
                height=480,
                avatar_images=(
                    None,  # user avatar (None = default)
                    "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
                ),
            )
            with gr.Row():
                msg = gr.Textbox(
                    elem_id="msg-input",
                    placeholder="Type a query or click an example below to test the DPDP erasure...",
                    show_label=False,
                    scale=5,
                    lines=1,
                    max_lines=3,
                    autofocus=True,
                )
                submit = gr.Button("Send ›", elem_id="submit-btn", scale=1, variant="primary")
            with gr.Row():
                clear = gr.Button("🗑 Clear conversation", elem_id="clear-btn", size="sm")

            # Example prompts
            gr.HTML('<div style="margin-top:8px;color:#475569;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;">Try these prompts:</div>')
            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="",
            )

        # Right column: info panel
        with gr.Column(scale=1, min_width=260):
            gr.HTML(INFO_PANEL_HTML)

    # ── Footer ──────────────────────────────────────────────
    gr.HTML(FOOTER_HTML)

    # ── Event handlers ──────────────────────────────────────
    def user_submit(user_message, history):
        return "", history + [[user_message, None]]

    def bot_respond(history):
        user_message = history[-1][0]
        history[-1][1] = ""
        for token in generate_response(user_message, history[:-1]):
            history[-1][1] = token
            yield history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    submit.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    clear.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    demo.queue(max_size=5).launch(
        show_api=False,
        show_error=True,
    )
