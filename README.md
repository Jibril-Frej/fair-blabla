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

## Datasets

| Dataset | Test file | Rows | Length (mean / median) | Type of text | What is annotated |
|---|---|---|---|---|---|
| [StereoDetect](https://aclanthology.org/2025.findings-emnlp.216/) | `stereodetect/test.csv` | 1,738 | 9.6 / 7 | Isolated short sentences | `labels`: 0 anti-stereotype, 1 stereotype, 2 neutral without target group, 3 neutral with target group, 4 bias; plus category and target group |
| [SBIC](https://aclanthology.org/2020.acl-main.486/) | `sbic/SBIC.v2.agg.tst.csv` | 4,691 | 20.2 / 18 | Social-media posts (Twitter, Reddit, Gab, Stormfront): informal, often offensive | Offensiveness, intent, lewdness, targeted group and free-text implied stereotype. Note `hasBiasedImplication` is inverted: **0 = biased implication** (1,924), 1 = none (2,767) |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) | `crows_pairs/crows_pairs_anonymized.csv` (no split) | 1,508 pairs | 13.1 / 12 | Minimal pairs of isolated sentences differing only by the group mentioned | Which sentence is more stereotypical (`sent_more`/`sent_less`), `stereo`/`antistereo` direction, 9 bias types (incl. socioeconomic, disability, age) |
| [ToxiGen](https://aclanthology.org/2022.acl-long.234/) | `toxigen/annotated_test.csv` | 940 | 18.6 / 17 | Isolated GPT-3-generated statements about a minority group, social-media style | Human toxicity 1–5 (`toxicity_human`, averaged over 3 annotators), intent, positive stereotyping, framing, 13 target groups |

## Evaluation samples

Draw 50 random rows (seed 42 by default) from each test set above into `data/samples/<dataset>.csv`:

```bash
python scripts/sample_test_sets.py
```

Options: `--n <rows>`, `--seed <seed>`, `--data <dir>`. 

## Bias detection models

Models that output the types of bias present in the text.

### Linguistic indicators of stereotypes

- **Paper:** Görge, Mock & Allende-Cid, *Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models*, FAccT 2025 — [ACM DL](https://dl.acm.org/doi/10.1145/3715275.3732181), [arXiv](https://arxiv.org/abs/2502.19160)
- **How it works:** LLM + Classifier. LLM for linguistic indicator extraction + a linear regression provided by the authors combines the indicators into a stereotype-strength score.
- **Code:** [GitHub](https://github.com/r-goerge/Detecting-Linguistic-Indicators-for-Stereotype-Assessment-with-LLMs) (Apache-2.0), with prompts and the regression model.
- **Output:** social category targeted, per-indicator labels, graded stereotype score, explanation.

### Demographic-axis prompting with retrieved examples

- **Paper:** Majumdar, Chen, Li & Wang, *Evaluating LLMs for Detecting Demographic-Targeted Social Bias: A Comprehensive Benchmark Study*, 2nd Workshop on Identity-Aware AI, 2026 — [ACL Anthology](https://aclanthology.org/2026.iaai-1.5/), [arXiv](https://arxiv.org/abs/2510.04641) (workshop paper)
- **How it works:** RAG style. Retrieve 5 most similar examples for each category (gender & sexual identity, sexual orientation, disability, age, race & ethnicity, nationality, religion, socio-economic status, physical appearance). Then an LLM (based on the examples) takes the decision.
- **Code:** not found; the prompt is given in the paper's appendix (Figure 3).
- **Output:** categories of bias found.

### Granite Guardian with custom bias criteria

- **Paper:** Padhi et al., *Granite Guardian: Comprehensive LLM Safeguarding*, NAACL 2025 Industry Track — [ACL Anthology](https://aclanthology.org/2025.naacl-industry.49/)
- **How it works:** Pure LLM. We define one criterion per bias type (gender, ethnicity, socio-economic status, ...) and, for each criterion, we ask the LLM to answer by yes or no if the input text is biased. We used the logits of the next token to get the probability of yes/no.
- **Code:** [Github](https://github.com/ibm-granite/granite-guardian)
- **Output:** yes/no + probability per bias type.

### BiasAlert-style retrieval-augmented judge

- **Paper:** Fan et al., *BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs*, EMNLP 2024 — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.820/), [arXiv](https://arxiv.org/abs/2407.10241)
- **How it works:** RAG style. Similar to Demographic-axis but with different sources for retrieval.
- **Code:** [GitHub](https://github.com/FanZT6/BiasAlert) (no license stated). It includes the bias database (`data/retrieval/bias_doc.tsv`), the retrieval scripts and the instruction template (`data/data_precessing/instruction_generation.py`).
- **Output:** biased yes/no, bias type, target group, biased description, explanation.

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

Each method predicts a set of bias types per text; an empty set means "not biased". 
We report the micro-F1 of these sets against the gold types.

<!-- results:start -->
| Method | Model | StereoDetect | SBIC | CrowS-Pairs | ToxiGen | Average |
|---|---|---:|---:|---:|---:|---:|
| Linguistic indicators (Görge et al.) | Qwen3.8-27B (FP8) | 0.40 | 0.34 | 0.60 | 0.29 | 0.41 |
| Demographic axes, 5-shot (Majumdar et al.) | Qwen3.8-27B (FP8) | 0.64 | 0.80 | 0.74 | 0.62 | 0.70 |
| Guardian per-type criteria (Padhi et al.) | Qwen3.8-27B (FP8) | 0.43 | 0.62 | 0.49 | 0.49 | 0.51 |
| BiasAlert-style RAG (Fan et al.) | Qwen3.8-27B (FP8) | 0.33 | 0.67 | 0.68 | 0.67 | 0.59 |
<!-- results:end -->
