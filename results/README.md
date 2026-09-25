# Results: bias detection methods with Qwen3.8-27B and Granite Guardian

Run on 2026-09-25 on the CYD GPU cluster, one NVIDIA H200 (141 GB) per model, with the vLLM
OpenAI-compatible server `vllm/vllm-openai:v0.30.0`
(`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`) in Apptainer.

## Models

| Results dir | Weights | Precision | Used for |
|---|---|---|---|
| `qwen3.8-27b-bf16` | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | BF16 (not quantized) | all four methods |
| `qwen3.8-27b-fp8` | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8 (official) | all four methods; control for the uncensored model, which only exists in FP8 |
| `qwen3.8-27b-uncensored-fp8` | [`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) | FP8 | all four methods |
| `granite-guardian-3.3-8b` | [`ibm-granite/granite-guardian-3.3-8b`](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b) | BF16 | Guardian method (per-type criteria and built-in `social_bias`) |

Decoding: greedy (temperature 0, top_p 1, seed 0). Qwen runs with thinking off
(`enable_thinking=False` in the chat template). Granite Guardian runs with `think=False`.

## Files

- `<model>/<method>/<dataset>.jsonl`: one line per item. It holds the gold labels (`gold_biased`, `gold_types`,
  `gold_score`), the prediction (`pred_biased`, `score`, `types`), `parse_ok` and the raw model output.
- `<model>/run.json`: the model and the settings.
- `retrieval/demographic_axes.jsonl`, `retrieval/biasalert.jsonl`: the ids of the retrieved examples and references
  per item (top 10; the methods use the top 5).
- `metrics.csv`: every metric per method, model and dataset (see `src/fairblabla/evaluate.py`).

## Gold labels

| Dataset | Biased (binary) | Types |
|---|---|---|
| EMGSD | `category == stereotype` | `stereotype_type` |
| StereoDetect | labels 1 (stereotype) and 4 (bias); 0, 2, 3 are not biased | `Category` |
| FSB | graded only (`score_normalized`) | gender |
| SBIC | `hasBiasedImplication == 0` | `targetCategory`, refined with `targetMinority` |
| CrowS-Pairs | pairs: the stereotypical sentence is `sent_more` for `stereo` rows and `sent_less` for `antistereo` rows | `bias_type` |
| ToxiGen | mean human toxicity ≥ 3 (1–5 scale) | `target_group` |
| GUS | any `STEREO` span | none |

All types are mapped to one shared list: gender, sexual orientation, disability, age, race/ethnicity, nationality,
religion, socio-economic, appearance, profession, other (`src/fairblabla/taxonomy.py`).

## Metrics

- **Main table (README):**
  - macro-F1 of the binary decision (EMGSD, StereoDetect, SBIC, ToxiGen, GUS);
  - Spearman ρ between the method's score and the human score (FSB);
  - pairwise accuracy (CrowS-Pairs): the share of pairs where the stereotypical sentence gets the higher score, with ties counting 0.5.

  Answers that cannot be parsed count as "not biased". Their number is in `metrics.csv` (`n_parse_fail`).
- **Type hit rate:** computed on gold-biased items whose gold type is known (for CrowS, the stereotypical sentence). It is the share of these items where the method flags the text *and* one of its predicted types is a gold type.
- `metrics.csv` also has AUROC, and Spearman for ToxiGen.
- **n = 50 per dataset** (100 sentences for CrowS-Pairs), so differences of a few points are within noise. For a proportion near 0.5, the 95% interval is about ±0.14.

## Adaptations to the published methods

- **Linguistic indicators (Görge et al.):**
  - Prompt P_F_01 and the SCSC regression come from the authors' repository. Their weights are copied as numbers into `linguistic_indicators.py`.
  - The JSON is parsed leniently.
  - A text is called biased when the SCSC score is above 0.5.
  - A second call maps the category label to a bias type.
  - The prompt only asks about gender and race, as in the paper.
- **Demographic axes (Majumdar et al.):**
  - The policy and the prompt come from the paper (Fig. 3).
  - The 5 nearest examples (BGE-M3) come from the train splits of EMGSD, StereoDetect, SBIC and ToxiGen, with texts found in the samples removed.
  - The paper does not show how examples are formatted, so we format them ourselves.
  - The output is binary, so CrowS-Pairs pairs often tie.
- **Guardian per-type criteria (Padhi et al.):**
  - One custom criterion per type (10 types), identical for Granite Guardian (through its chat template) and for Qwen (plain yes/no judge prompt).
  - P(yes) comes from the token logprobs.
  - The built-in `social_bias` risk is only run with Granite Guardian.
- **BiasAlert-style RAG (Fan et al.):**
  - The judge is the off-the-shelf model, not the authors' fine-tuned Llama-2 (its weights are not released).
  - The 5 nearest entries of their bias database (contriever-msmarco) are inserted as "References" between their instruction and the sentence.
  - Their instruction ends with "Let's think step by step", so Qwen reasons before the answer template (even with
    thinking off). We allow 4096 new tokens, take the **last** "Yes/No, the following SENTENCE is …" as the verdict,
    and the score is P("Yes") at that verdict token. Runs with 150 and 1024 tokens cut many answers off and were discarded.
  - The database is built from SBIC, so the SBIC scores are contaminated.
