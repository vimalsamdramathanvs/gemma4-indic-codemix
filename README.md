# Gemma 4 for Code-Mixed Hindi

Fine-tuning Google's Gemma 4 E2B with LoRA (Unsloth, free Colab T4) to convert between
English, native-script code-mixed Hindi, and casual romanized Hindi.
HacktoberFest HackDay, Theme 3, Problem 3A.

[Open the notebook in Colab](https://colab.research.google.com/github/vimalsamdramathanvs/gemma4-indic-codemix/blob/main/gemma4_finetune.ipynb)
| Demo video: TBD (Drive or YouTube link)

## Problem
People mix English with regional languages in chat, with informal spelling and
transliteration. General models handle this inconsistently. We fine-tune Gemma to do
four Hindi conversions and measure whether it beats the plain model.

## Tasks
| Task | Input | Output |
|---|---|---|
| en_to_roman | English | Casual romanized Hindi |
| en_to_mixed | English | Hindi-English code-mixed, native script |
| roman_to_en | Casual romanized Hindi | English |
| mixed_to_roman | Code-mixed, native script | Casual romanized Hindi |

## Dataset
- ai4bharat/IndicCMix, Hindi file (hi.parquet), gated dataset, about 104,800 sentences.
- Split by sentence BEFORE creating the four tasks, so no test sentence appears in
  training in any direction.
- Test: 10 held-out sentences x 4 tasks = 40 examples.
- Train: 2,400 examples from the remaining sentences (about 400 seen in 100 steps).

## Method
| Setting | Value |
|---|---|
| Base model | unsloth/gemma-4-E2B-it, 4-bit |
| LoRA | rank 8, alpha 8, language layers |
| Steps / batch | 100 steps, effective batch 4 |
| Learning rate | 2e-4 |
| Seed | 3407 |
| Training time | TBD |
| Metric | chrF (sacrebleu), same test set and settings for every model |

## Results
| Task | Plain | Prompted (3 examples) | Fine-tuned |
|---|---|---|---|
| en_to_mixed | 29.7 | TBD | TBD |
| en_to_roman | 44.1 | TBD | TBD |
| mixed_to_roman | 45.3 | TBD | TBD |
| roman_to_en | 47.6 | TBD | TBD |
| **Average** | **41.7** | TBD | TBD |

Human naturalness rating (blind, 1 to 5, 20 outputs per model): TBD

### Where fine-tuning helped, and where it didn't
TBD: two or three honest sentences on which tasks improved, which didn't, and why.

## Example outputs
| Input | Reference | Plain | Fine-tuned |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

Include at least one case where the fine-tuned model is wrong.

## Error analysis
Lowest-scoring 25 fine-tuned outputs, one label each:
| Error type | Count |
|---|---|
| Valid spelling variant | TBD |
| Dropped or added words | TBD |
| Wrong script | TBD |
| Extra explanation | TBD |
| Meaning error | TBD |

## Application
A custom web app (FastAPI backend, hand-written HTML, CSS and JavaScript frontend)
that runs inside the Colab session after training. A simpler Gradio version is also included.

| File | What it does |
|---|---|
| [app/custom_frontend.py](app/custom_frontend.py) | Backend API and the web page |
| [app/speed_patch.py](app/speed_patch.py) | Shorter outputs and cached evidence results |
| [app/look_upgrade.py](app/look_upgrade.py) | Visual styling only |
| [app/gradio_app.py](app/gradio_app.py) | Simple Gradio fallback |

Features:
1. **Playground.** Compare plain and fine-tuned output side by side, with timings,
   copy buttons, history and a dark/light toggle.
2. **Auto-detect task.** Detects the script and common romanized Hindi words as you
   type. This is a simple rule-based check, not a trained model.
3. **Live chrF.** Type your own expected answer and both models are scored.
4. **Evidence tab.** Browse held-out test sentences and see the reference, both outputs
   and chrF. These reuse the outputs from the evaluation, so they load instantly.
5. **Results tab.** Per-task chart from the measured scores.

Speed: the speed patch caps live outputs at 60 tokens, switches Unsloth to inference
mode and caches evidence results. Measured live latency on a T4: TBD seconds per
comparison. Evidence-tab results are the same 100-token outputs used in the results table.

![Playground](screenshots/app_playground.png)
![Evidence](screenshots/app_evidence.png)
![Results](screenshots/app_results.png)

## How to run
1. Open the notebook in Colab, select Runtime, Change runtime type, T4 GPU.
2. Make a Hugging Face account, accept the IndicCMix terms, create a read token.
3. Add it in Colab Secrets as HF_TOKEN with Notebook access on.
4. Run the cells in order. Skip the starter's original Step 3, 5 and 6 cells (sample IT-ticket code).
5. Run the app cells last. The public link is temporary and works only while the session runs.

## Limitations
- 40 test examples, so differences of a few chrF points are noise.
- chrF measures character overlap with one reference, so valid alternative spellings score low.
- Only 100 training steps on a subset of the data.
- Live outputs in the app are capped at 60 tokens, so long sentences may be cut off.
- Auto-detect is a heuristic and can misjudge short text.
- Hindi only.

## Credits
Dataset: AI4Bharat IndicCMix. Model: Google Gemma 4. Training: Unsloth.
