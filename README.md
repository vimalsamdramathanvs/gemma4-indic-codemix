# Gemma 4 for Code-Mixed Hindi (Indic Code-Mix)

Fine-tuning Google's Gemma 4 E2B with LoRA (Unsloth, free Colab T4 GPU) to convert
between English, native-script code-mixed text, and casual romanized text.
Built for HacktoberFest HackDay, Theme 3, Problem 3A.

[Open the notebook in Colab](https://colab.research.google.com/github/vimalsamdramathanvs/gemma4-indic-codemix/blob/main/gemma4_finetune.ipynb)

## Problem
People mix English with regional languages when chatting, using informal spelling,
transliteration and mixed scripts. General models handle this inconsistently.
This project teaches Gemma to do four conversions in Hindi.

## Tasks
| Task | Input | Output |
|---|---|---|
| en_to_roman | English | Casual romanized Hindi |
| en_to_mixed | English | Hindi-English code-mixed, native script |
| roman_to_en | Casual romanized Hindi | English |
| mixed_to_roman | Code-mixed, native script | Casual romanized Hindi |

## Dataset
- Source: ai4bharat/IndicCMix, Hindi file (hi.parquet), about 104,800 sentences. Gated dataset.
- Each sentence exists in four forms: English, native-script code-mixed, full native script, romanized casual.
- Split by sentence BEFORE expanding into tasks, so no test sentence appears in training in any direction.
- Test set: 10 held-out sentences x 4 tasks = 40 examples.
- Training set: 2,400 examples drawn from the remaining sentences.

## Method
- Base model: unsloth/gemma-4-E2B-it, 4-bit quantized
- LoRA rank 8, alpha 8, language layers (attention and MLP)
- 100 training steps, learning rate 2e-4, effective batch size 4, seed 3407
- Training time: TBD minutes
- Metric: chrF (sacrebleu), same test set and settings for both models

## Results
| Task | Plain Gemma (chrF) | Fine-tuned (chrF) | Change |
|---|---|---|---|
| en_to_mixed | 29.7 | TBD | TBD |
| en_to_roman | 44.1 | TBD | TBD |
| mixed_to_roman | 45.3 | TBD | TBD |
| roman_to_en | 47.6 | TBD | TBD |
| **Average** | **41.7** | **TBD** | **TBD** |

Human naturalness rating (1 to 5, blind, 20 outputs per model): TBD

## Example outputs
TBD: add 3 to 5 examples, including at least one failure.

## Application
A Gradio app shows the plain and fine-tuned models side by side for the same input,
with a task dropdown. It runs inside Colab (see the notebook, last cells).

## How to run
1. Open the notebook in Colab and select Runtime, Change runtime type, T4 GPU.
2. Create a Hugging Face account, accept the IndicCMix terms, make a read token.
3. Add the token in Colab Secrets as HF_TOKEN with Notebook access on.
4. Run the cells in order. Do not run the original Step 3, 5 and 6 cells; they use the starter's sample code.

## Limitations
- The test set is small (40 examples), so differences of a few chrF points are noise.
- chrF measures character overlap with one reference, so a natural alternative phrasing scores low.
- Only 100 training steps, so about 400 examples were seen.
- Hindi only so far.

## Credits
Dataset: AI4Bharat IndicCMix. Model: Google Gemma 4. Training: Unsloth.
