# fair-blabla

## Download the datasets

Requires Python ≥ 3.12 .
Run from the repository root; each script downloads one dataset into `data/<name>/`
and prints its size and columns.

Download everything:

```bash
python scripts/download/all.py
```

Or one dataset at a time:

```bash
python scripts/download/stereodetect.py
python scripts/download/sbic.py
python scripts/download/crows_pairs.py
python scripts/download/toxigen.py
```

Use `--out <dir>` to change the output directory. Already-downloaded files are skipped.

## Datasets (test sets)

Lengths are in words (whitespace tokens). Datasets without an official split are used in full.

| Dataset | Test file | Rows | Length (mean / median) | Type of text | What is annotated |
|---|---|---|---|---|---|
| [StereoDetect](https://aclanthology.org/2025.findings-emnlp.216/) | `stereodetect/test.csv` | 1,738 | 9.6 / 7 | Isolated short sentences | `labels`: 0 anti-stereotype, 1 stereotype, 2 neutral without target group, 3 neutral with target group, 4 bias; plus category and target group |
| [SBIC](https://aclanthology.org/2020.acl-main.486/) | `sbic/SBIC.v2.agg.tst.csv` | 4,691 | 20.2 / 18 | Social-media posts (Twitter, Reddit, Gab, Stormfront): informal, often offensive | Offensiveness, intent, lewdness, targeted group and free-text implied stereotype. Note `hasBiasedImplication` is inverted: **0 = biased implication** (1,924), 1 = none (2,767) |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) | `crows_pairs/crows_pairs_anonymized.csv` (no split) | 1,508 pairs | 13.1 / 12 | Minimal pairs of isolated sentences differing only by the group mentioned | Which sentence is more stereotypical (`sent_more`/`sent_less`), `stereo`/`antistereo` direction, 9 bias types (incl. socioeconomic, disability, age) |
| [ToxiGen](https://aclanthology.org/2022.acl-long.234/) | `toxigen/annotated_test.csv` | 940 | 18.6 / 17 | Isolated GPT-3-generated statements about a minority group, social-media style | Human toxicity 1–5 (`toxicity_human`, averaged over 3 annotators), intent, positive stereotyping, framing, 13 target groups |

## Evaluation samples

Draw 50 random rows (seed 42) from each test set above into `data/samples/<dataset>.csv`:

```bash
python scripts/sample_test_sets.py
```

Options: `--n <rows>`, `--seed <seed>`, `--data <dir>`. The output is identical across runs for the same seed.

## Bias detection models

Methods we run. All are training-free (prompting / in-context examples on open-weight LLMs, no fine-tuning on our side) and output the types of bias found, not only biased / not biased.

### Linguistic indicators of stereotypes

- **Paper:** Görge, Mock & Allende-Cid, *Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models*, FAccT 2025 — [ACM DL](https://dl.acm.org/doi/10.1145/3715275.3732181), [arXiv](https://arxiv.org/abs/2502.19160)
- **How it works:** based on the Social Category and Stereotype Communication (SCSC) framework from social psychology. A few-shot prompted LLM detects linguistic indicators in a sentence: the social category mentioned (and how generic it is), the behaviour or feature attributed to it, the generalisation, and the explanation. A linear regression provided by the authors (already trained, no training on our side) combines the indicators into a stereotype-strength score.
- **Models:** evaluated with Llama-3.3-70B-Instruct, GPT-4, GPT-4o-mini, Mixtral-8x7B-Instruct and Llama-3.1-8B-Instruct (4-bit); Llama-3.3-70B-Instruct performs on par with GPT-4.
- **Code:** [GitHub](https://github.com/r-goerge/Detecting-Linguistic-Indicators-for-Stereotype-Assessment-with-LLMs) (Apache-2.0), with prompts and the regression model. It calls an OpenAI-compatible API, so it can point to a local vLLM server.
- **Output:** social category targeted, per-indicator labels, graded stereotype score, explanation.

### Demographic-axis prompting with retrieved examples

- **Paper:** Majumdar, Chen, Li & Wang, *Evaluating LLMs for Detecting Demographic-Targeted Social Bias: A Comprehensive Benchmark Study*, 2nd Workshop on Identity-Aware AI, 2026 — [ACL Anthology](https://aclanthology.org/2026.iaai-1.5/), [arXiv](https://arxiv.org/abs/2510.04641) (workshop paper)
- **How it works:** bias detection as multi-label classification over 9 axes: gender & sexual identity, sexual orientation, disability, age, race & ethnicity, nationality, religion, socio-economic status, physical appearance. The prompt is a "policy" defining each axis with biased and safe (e.g. anti-stereotype) examples. In the few-shot variant, the 5 or 10 most similar labelled examples (BGE-M3 embeddings, cosine similarity) are retrieved from a development pool and added to the prompt.
- **Models:** Llama-3.1-8B/70B, GLM-4-9B, Qwen-2.5-72B, Llama Guard-3-8B (prompting). Retriever: `BAAI/bge-m3`.
- **Code:** none found; the policy prompt is given in the paper's appendix (Figure 3).
- **Output:** biased / not biased + the list of axes targeted. No explanation or target group: to be added to the requested output format.

### Granite Guardian with custom bias criteria

- **Paper:** Padhi et al., *Granite Guardian: Comprehensive LLM Safeguarding*, NAACL 2025 Industry Track — [ACL Anthology](https://aclanthology.org/2025.naacl-industry.49/)
- **How it works:** an LLM trained by IBM to judge whether a text meets a risk criterion given in the prompt. Besides the built-in `social_bias` criterion, it accepts custom criteria written in natural language. We define one criterion per bias type (gender, ethnicity, socio-economic status, ...) and run one check per type; the probability of "yes" gives a score per type.
- **Models:** [`ibm-granite/granite-guardian-3.3-8b`](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b) (Apache-2.0); `think=True` adds a reasoning trace.
- **Output:** yes/no + probability per bias type (+ optional reasoning).
- **Note:** custom per-type criteria are our adaptation; the paper evaluates the built-in risks.

### BiasAlert-style retrieval-augmented judge

- **Paper:** Fan et al., *BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs*, EMNLP 2024 — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.820/), [arXiv](https://arxiv.org/abs/2407.10241)
- **How it works:** (1) a retriever fetches the 5 entries most similar to the input text from a database of ~41k known social biases (target group + biased description, built from SBIC and covering gender, race, culture, religion, social, disability, orientation); (2) an LLM reads the text and the retrieved entries and reasons step by step: identify the target group and the description, compare with the references, decide whether the text is biased.
- **Models in the paper:** retriever `facebook/contriever-msmarco`; detector Llama-2-7b-chat fine-tuned with LoRA on RedditBias. The fine-tuned weights are not released.
- **What we run:** the same retrieval pipeline with an off-the-shelf instruction-tuned LLM prompted with the paper's step-by-step instructions, without fine-tuning. This is our adaptation: results are not comparable with the paper's.
- **Code:** [GitHub](https://github.com/FanZT6/BiasAlert) (no license stated). It includes the bias database (`data/retrieval/bias_doc.tsv`), the retrieval scripts and the instruction template (`data/data_precessing/instruction_generation.py`).
- **Output:** biased yes/no, bias type, target group, biased description, explanation.
- **Note:** the database is built from SBIC, so SBIC results are contaminated and must be reported separately.

## Running the evaluation

The four methods run on the 50-row samples with open-weight models served by [vLLM](https://github.com/vllm-project/vllm) (OpenAI-compatible API).

```bash
python scripts/fetch_method_assets.py   # prompts and bias database from the authors' repositories -> data/methods/
python scripts/retrieve.py              # in-context examples (BGE-M3) and BiasAlert references (contriever-msmarco) -> results/retrieval/
python scripts/run_methods.py --base-url http://127.0.0.1:8000/v1 --model <served name> \
    --slug <results dir> --display "<name in the table>" --model-id <HF id> --kind chat --no-thinking \
    --methods linguistic_indicators demographic_axes guardian_criteria biasalert_rag
uv run python -m fairblabla.evaluate --readme README.md   # results/metrics.csv + the table below
```

On the Slurm cluster, `bash slurm/submit_all.sh` runs the retrieval step, then one job per model: each job starts a vLLM server on one H200 GPU and runs the methods against it (`slurm/serve_and_run.sbatch`). The vLLM image and the model weights are fetched first with `slurm/fetch_vllm_image.sh` and `slurm/download_models.sbatch`.

Outputs, per model: `results/<model>/<method>/<dataset>.jsonl` (one line per item: gold labels, prediction, predicted types, score, raw model output) and `results/<model>/run.json` (model and decoding settings). See [`results/README.md`](results/README.md) for the setup and the adaptations to each method.

## Results

Each method predicts a set of bias types per text; an empty set means "not biased". The cell is the
micro-F1 of these sets against the gold types, over all (text, type) pairs of the 50-row sample. An
unbiased text has an empty gold set, so any type predicted on it is a false positive; a missed gold type is a
false negative; an extra wrong type is a false positive. Only methods that output types are scored (Granite Guardian's
built-in `social_bias` is left out). Details in [`results/README.md`](results/README.md#metrics).

All methods run on Qwen3.8-27B-FP8 with structured outputs. We also ran Qwen3.8-27B in BF16, the uncensored
Qwen3.8-27B (FP8) and free-text answers. On each method, these variants give the same output on 81–100% of texts and
differ by at most 0.03 F1, within the 95% bootstrap interval of zero, so they are not shown here. The Guardian method
also ran on Granite Guardian 3.3 (8B), the model of the paper: it flags many types on most texts (type precision
0.14), for an average F1 of 0.24. All these scores are in [`results/metrics.csv`](results/metrics.csv).

<!-- results:start -->
| Method | Model | StereoDetect | SBIC | CrowS-Pairs | ToxiGen | Average |
|---|---|---:|---:|---:|---:|---:|
| Linguistic indicators (Görge et al.) | Qwen3.8-27B (FP8), structured | 0.40 | 0.34 | 0.60 | 0.29 | 0.41 |
| Demographic axes, 5-shot (Majumdar et al.) | Qwen3.8-27B (FP8), structured | 0.56 | 0.75 | 0.79 | 0.61 | 0.68 |
| Guardian per-type criteria (Padhi et al.) | Qwen3.8-27B (FP8), structured | 0.43 | 0.62 | 0.49 | 0.49 | 0.51 |
| BiasAlert-style RAG (Fan et al.) | Qwen3.8-27B (FP8), structured | 0.33 | 0.67 | 0.68 | 0.67 | 0.59 |
<!-- results:end -->
