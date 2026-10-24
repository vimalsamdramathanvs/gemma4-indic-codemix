import time

# 1. Faster inference mode (untested on Gemma 4; skipped safely if it fails)
try:
    FastModel.for_inference(model)
    print("inference mode on")
except Exception as e:
    print("for_inference skipped:", e)

# 2. Shorter outputs. Raise to 80 if answers look cut off.
FAST_TOKENS = 60

def gen_plain(prompt):
    with model.disable_adapter():
        return generate(prompt, max_new_tokens=FAST_TOKENS)

# 3. Replace the two old endpoints with faster ones
app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in ("/api/compare", "/api/evidence")]

@app.post("/api/compare")
def api_compare_fast(req: CompareReq):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Enter some text first."}, status_code=400)
    task, why = (detect(text) if req.task == "auto" or req.task not in TASKS else (req.task, "chosen by you"))
    prompt = f"{TASKS[task][2]}\n\nText: {text}"
    with GPU_LOCK:
        t = time.time(); f = generate(prompt, max_new_tokens=FAST_TOKENS); tf = time.time() - t
        t = time.time(); p = gen_plain(prompt); tp = time.time() - t
    out = {"task": task, "why": why, "plain": p, "tuned": f,
           "t_plain": round(tp, 1), "t_tuned": round(tf, 1)}
    if req.ref.strip():
        out["chrf_plain"], out["chrf_tuned"] = chrf(p, req.ref), chrf(f, req.ref)
    return out

@app.post("/api/evidence")
def api_evidence_fast(req: EvidenceReq):
    i = req.idx
    if not (0 <= i < len(test_examples)):
        return JSONResponse({"error": "Bad index"}, status_code=400)
    ex = test_examples[i]
    src = ex["input"].split("\n\nText: ", 1)[1]
    try:   # instant: reuse the outputs already evaluated
        b, f = baseline_df.iloc[i], finetuned_df.iloc[i]
        return {"task": ex["task"], "src": src, "ref": ex["output"],
                "plain": b["plain_model"], "tuned": f["fine_tuned_model"],
                "chrf_plain": round(float(b["chrf"]), 1), "chrf_tuned": round(float(f["chrf"]), 1),
                "t_plain": 0, "t_tuned": 0}
    except Exception:   # fallback if those tables are not in memory
        with GPU_LOCK:
            f = generate(ex["input"], max_new_tokens=FAST_TOKENS); p = gen_plain(ex["input"])
        return {"task": ex["task"], "src": src, "ref": ex["output"], "plain": p, "tuned": f,
                "chrf_plain": chrf(p, ex["output"]), "chrf_tuned": chrf(f, ex["output"]),
                "t_plain": 0, "t_tuned": 0}

# 4. Warm-up so the first real click isn't slow
t = time.time(); generate("Translate to Hindi: hello", max_new_tokens=10)
print(f"warm-up done in {time.time()-t:.1f}s. Refresh the app page now.")
