import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔧 Models to compare (all ~1.5–2B parameters)
MODEL_IDS = {
    "SmolLM-1.7B": "HuggingFaceTB/SmolLM-1.7B",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

SYSTEM_PROMPT = (
    "You are an assistant that helps users with Excel formulas. "
    "When the user asks for a formula: output ONLY the Excel formula if possible. "
    "When asked a conceptual question about Excel: respond briefly and directly."
)

def build_prompt(user_text: str) -> str:
    # Simple, model-agnostic prompt (no special chat template)
    return f"{SYSTEM_PROMPT}\n\nUser: {user_text}\nAssistant:"

# Load all models + tokenizers at startup
MODELS = {}

def load_models():
    global MODELS
    for name, model_id in MODEL_IDS.items():
        print(f"Loading {name} from {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",      # use GPU if available, otherwise CPU
            torch_dtype="auto",     # let HF pick best dtype
        )
        MODELS[name] = (tok, mdl)

load_models()

def generate_all(user_text: str, temperature: float, max_new_tokens: int):
    """
    Run the same Excel prompt through all three models and return three outputs.
    """
    if not user_text or not user_text.strip():
        return "", "", ""

    temperature = float(temperature)
    max_new_tokens = int(max_new_tokens)
    prompt = build_prompt(user_text.strip())

    outputs = {}
    for name, (tokenizer, model) in MODELS.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=0.95 if temperature > 0.0 else 1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        # Trim prompt prefix if model echoes it
        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[-1].strip()
        outputs[name] = text.strip()

    return (
        outputs.get("SmolLM-1.7B", ""),
        outputs.get("Qwen2.5-1.5B-Instruct", ""),
        outputs.get("DeepSeek-R1", ""),
    )

with gr.Blocks(title="6SigmaMind – Small Excel Models Benchmark") as demo:
    gr.Markdown(
        """
        # 🧠 6SigmaMind — Small Excel Models Benchmark

        Type an Excel-related question or task in natural language.  
        All three models will answer in parallel:

        - **SmolLM-1.7B** (your 6SigmaMind baseline)
        - **Qwen2.5-1.5B-Instruct**
        - **DeepSeek-R1**

        Try prompts like:
        - "Write an Excel formula that sums values in column C where column B is 'Closed'."
        - "Give an Excel formula that returns the last non-empty value from column B."
        - "Calculate the standard deviation of B2:B80 in Excel."
        - "Return the correlation between columns A and B."
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="Your Excel task or question",
                placeholder="e.g., Sum C where B equals 'Closed'",
                lines=3,
            )

            with gr.Row():
                temperature = gr.Slider(
                    0.0, 1.0, value=0.1, step=0.05, label="Temperature"
                )
                max_tokens = gr.Slider(
                    16, 256, value=96, step=8, label="Max new tokens"
                )

            gr.Examples(
                examples=[
                    ["Write an Excel formula that sums values in column C where column B is \"Closed\"."],
                    ["Return the price in column D where the SKU in column A matches H2."],
                    ["Calculate the correlation between columns A and B in Excel."],
                    ["Give an Excel formula that counts how many cells in A2:A100 are greater than 50."],
                    ["Give an Excel formula that returns the last non-empty value in column B."],
                ],
                inputs=[user_input],
                label="Example prompts",
            )

            run_btn = gr.Button("Generate with all three models")

        with gr.Column(scale=3):
            with gr.Row():
                out_smol = gr.Textbox(
                    label="SmolLM-1.7B output",
                    lines=8,
                    show_copy_button=True,
                )
                out_qwen = gr.Textbox(
                    label="Qwen2.5-1.5B-Instruct output",
                    lines=8,
                    show_copy_button=True,
                )
                out_gemma = gr.Textbox(
                    label="DeepSeek-R1 output",
                    lines=8,
                    show_copy_button=True,
                )

    run_btn.click(
        fn=generate_all,
        inputs=[user_input, temperature, max_tokens],
        outputs=[out_smol, out_qwen, out_gemma],
    )

if __name__ == "__main__":
    demo.launch()
