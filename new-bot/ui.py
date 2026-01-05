import gradio as gr
from bot import run_once   # we will refactor run()

def run_bot(symbol):
    result = run_once(symbol)
    return result

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 LLM-Gated Crypto Trading Bot")

    symbol_input = gr.Textbox(
        label="Trading Symbol",
        value="BNB/USDT"
    )

    run_btn = gr.Button("Run Analysis")

    output = gr.JSON(label="Bot Output")

    run_btn.click(
        fn=run_bot,
        inputs=symbol_input,
        outputs=output
    )

demo.launch()
