"""Registry of the bias detection methods: name -> (load(**ctx) -> state, predict(item, client, state) -> dict).

predict returns at least: pred_biased (1/0, None if the answer could not be parsed), score (float,
higher = more biased), types (list of shared types), parse_ok, raw (model output).
"""
from . import biasalert_rag, demographic_axes, guardian_criteria, linguistic_indicators

METHODS = {
    linguistic_indicators.NAME: (linguistic_indicators.load, linguistic_indicators.predict),
    demographic_axes.NAME: (demographic_axes.load, demographic_axes.predict),
    guardian_criteria.NAME: (guardian_criteria.load, guardian_criteria.predict),
    guardian_criteria.BUILTIN_NAME: (guardian_criteria.load, guardian_criteria.predict_builtin),
    biasalert_rag.NAME: (biasalert_rag.load, biasalert_rag.predict),
}

LABELS = {
    linguistic_indicators.NAME: "Linguistic indicators (Görge et al.)",
    demographic_axes.NAME: "Demographic axes, 5-shot (Majumdar et al.)",
    guardian_criteria.NAME: "Guardian per-type criteria (Padhi et al.)",
    guardian_criteria.BUILTIN_NAME: "Guardian built-in social_bias (Padhi et al.)",
    biasalert_rag.NAME: "BiasAlert-style RAG (Fan et al.)",
}
