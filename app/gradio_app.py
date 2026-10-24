# Runs inside colab notebook after traning. Needs model, generate, TASKS and test_examples in memory.
import gradio as gr, re, sacrebleu
gr.close_all()

HINDI_WORDS = {"hai","hain","ke","ki","ka","mein","ko","se","par","aur","nahi","kya",
               "tha","thi","liye","hota","hote","bhi","yeh","woh","jo","toh","apna"}

def detect(text):
    if re.search(r"[\u0900-\u097F]", text):
        return "mixed_to_roman", "Devanagari script found"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if len(words & HINDI_WORDS) >= 2:
        return "roman_to_en", "romanized Hindi words found"
    return "en_to_roman", "looks like English"

def gen_plain(prompt):
    with model.disable_adapter():
        return generate(prompt, max_new_tokens=100)

def chrf(a, b):
    return sacrebleu.sentence_chrf(a, [b]).score

def run(text, task, auto, ref):
    if not text.strip():
        raise gr.Error("Enter some text first.")
    note = ""
    if auto:
        task, why = detect(text)
        note = f"Auto-selected task: {task} ({why})"
    prompt = f"{TASKS[task][2]}\n\nText: {text}"
    p, f = gen_plain(prompt), generate(prompt, max_new_tokens=100)
    if ref.strip():
        note += f"\nchrF vs your reference: plain {chrf(p, ref):.1f} | fine-tuned {chrf(f, ref):.1f}"
    return p, f, note

labels = [f"{i}: {ex['task']}" for i, ex in enumerate(test_examples)]

def evidence(label):
    ex = test_examples[int(label.split(":")[0])]
    p, f = gen_plain(ex["input"]), generate(ex["input"], max_new_tokens=100)
    src = ex["input"].split("\n\nText: ", 1)[1]
    return src, ex["output"], p, f, f"chrF  plain {chrf(p, ex['output']):.1f} | fine-tuned {chrf(f, ex['output']):.1f}"

with gr.Blocks(title="Code-Mixed Hindi with Gemma 4") as demo:
    gr.Markdown("# Code-Mixed Hindi with Gemma 4\nPlain vs fine-tuned, same input.")
    with gr.Tab("Try it"):
        auto = gr.Checkbox(value=True, label="Auto-detect task from the text")
        task = gr.Dropdown(list(TASKS.keys()), value="en_to_roman", label="Task (used if auto is off)")
        box = gr.Textbox(label="Input text", lines=3)
        ref = gr.Textbox(label="Optional: your expected answer (for live chrF)", lines=2)
        btn = gr.Button("Compare", variant="primary")
        with gr.Row():
            a = gr.Textbox(label="Plain Gemma", lines=5)
            b = gr.Textbox(label="Fine-tuned Gemma", lines=5)
        info = gr.Textbox(label="Info", lines=3)
        btn.click(run, [box, task, auto, ref], [a, b, info])
    with gr.Tab("Evidence (held-out test set)"):
        gr.Markdown("These sentences were never used in training.")
        pick = gr.Dropdown(labels, value=labels[0], label="Test example")
        go = gr.Button("Run", variant="primary")
        s = gr.Textbox(label="Input"); r = gr.Textbox(label="Reference answer")
        with gr.Row():
            pa = gr.Textbox(label="Plain Gemma", lines=4); fb = gr.Textbox(label="Fine-tuned Gemma", lines=4)
        sc = gr.Textbox(label="Scores")
        go.click(evidence, pick, [s, r, pa, fb, sc])

demo.launch(share=True)
