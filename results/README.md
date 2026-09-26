# Results: bias detection methods with Qwen3.8-27B and Granite Guardian

Run on 2026-09-25 on the CYD GPU cluster, one NVIDIA H200 (141 GB) per model, with the vLLM
OpenAI-compatible server `vllm/vllm-openai:v0.30.0`
(`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`) in Apptainer.

## Models

| Results dir | Weights | Precision | Used for |
|---|---|---|---|
| `qwen3.8-27b-bf16` | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | BF16 (not quantized) | all four methods |
| `qwen3.8-27b-fp8` | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8 (official) | all four methods; control for the uncensored model, which only exists in FP8 |
| `qwen3.8-27b-uncensored-fp8` | [`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) (revision `0f3cdb8`) | FP8 | all four methods |
| `granite-guardian-3.3-8b` | [`ibm-granite/granite-guardian-3.3-8b`](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b) | BF16 | Guardian method (per-type criteria and built-in `social_bias`) |

Decoding: greedy (temperature 0, top_p 1, seed 0). Qwen runs with thinking off
(`enable_thinking=False` in the chat template). Granite Guardian runs with `think=False`.

### Free-text and structured runs

Each Qwen model is run twice:

- **free text** (`qwen3.8-27b-bf16`, `qwen3.8-27b-fp8`): the answer formats of the papers, parsed with regular expressions.
- **structured** (`*-structured`): the same prompts, with the answer constrained by vLLM structured outputs
  (xgrammar), so answers follow the expected format:
  - demographic axes: regex `S10|S[1-9](,S[1-9])*`;
  - Guardian criteria (Qwen): choice `yes` / `no`;
  - linguistic indicators: JSON schema with the prompt's keys and allowed values, and a regex for the type answer;
  - BiasAlert: JSON with `reasoning` (at most 2000 characters) first, then the TEMPLATE slots `biased` (`Yes`/`No`),
    `bias_type`, `social_group`, `social_attribute`. The score is P("Yes") at the `biased` token.

  Granite Guardian keeps its own output format and is not constrained.

  The JSON grammar still allows whitespace between tokens: vLLM 0.30 ignores the per-request
  `disable_any_whitespace` option with the xgrammar backend. In one structured BiasAlert answer
  (uncensored model, EMGSD) the model repeated `\r` until it reached the token limit. It counts as unparsed.

Parse failures over all runs: 2 out of 7,800 answers. Besides the one above, a free-text BiasAlert answer
(uncensored model, CrowS-Pairs) was still reasoning at the 4096-token limit.

Thinking mode is off in all runs. Only BiasAlert produces reasoning, because its instruction asks for it
("Let's think step by step"). Only the verdict is scored, never the reasoning.

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
| SBIC | `hasBiasedImplication == 0` | `targetCategory`, refined with `targetMinority` |
| CrowS-Pairs | pairs: the stereotypical sentence is `sent_more` for `stereo` rows and `sent_less` for `antistereo` rows | `bias_type` |
| ToxiGen | mean human toxicity ≥ 3 (1–5 scale) | `target_group` |

All types are mapped to one shared list: gender, sexual orientation, disability, age, race/ethnicity, nationality,
religion, socio-economic, appearance, profession, other (`src/fairblabla/taxonomy.py`).

## Metrics

- **Table in the main README: bias type F1.** For each text, the gold set is its gold types if it is biased and the
  empty set if it is not. The predicted set is the method's types if it flags the text, and empty otherwise. Each
  method has a "no bias" answer:
  - demographic axes: `S10` ("safe");
  - Guardian criteria: P(yes) ≤ 0.5 for all 10 types;
  - BiasAlert: the "No" verdict;
  - linguistic indicators: SCSC score ≤ 0.5.

  True positives, false positives and false negatives are counted over all (text, type) pairs of a dataset, and the
  cell is the micro-F1. So a type predicted on an unbiased text, or an extra wrong type on a biased text, is a
  false positive, and a missed gold type is a false negative.
  - The type `other` is dropped from both sets. Biased texts with no known gold type (some SBIC posts) are skipped.
  - CrowS-Pairs: only the stereotypical sentence of each pair is scored. The other sentence is only *less*
    stereotypical, not labelled unbiased.
  - Left out: Granite Guardian's built-in `social_bias` (no types).
  - Some gold types are out of reach for some methods. For example, demographic axes has no `profession` axis
    (EMGSD, StereoDetect), and linguistic indicators only asks about gender and race.

  Answers that cannot be parsed count as "not biased". Their number is in `metrics.csv` (`n_parse_fail`).
- `metrics.csv` also has type precision and recall (`type_precision`, `type_recall`, `n_type` = texts scored). It
  also has the binary detection metrics: macro-F1 and AUROC of biased vs not biased, Spearman ρ with the human
  score (ToxiGen), and CrowS-Pairs pairwise accuracy (the share of pairs where the stereotypical sentence gets
  the higher score, ties 0.5).
- **n = 50 per dataset** (50 stereotypical sentences for CrowS-Pairs), so differences of a few points are within
  noise.

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
