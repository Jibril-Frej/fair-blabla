"""Loaded by Python at start-up when slurm/compat is on PYTHONPATH (Mistral server only, see
slurm/exercises_all.sh). vLLM 0.30's pixtral.py imports two names that transformers 5.17 no longer
has: PixtralRotaryEmbedding (renamed PixtralVisionRotaryEmbedding) and position_ids_in_meshgrid
(removed). Both are only used by PixtralHFVisionModel, the Hugging Face-format image encoder, which
a Mistral-format model does not use (and no images are sent)."""
try:
    import transformers.models.pixtral.modeling_pixtral as pixtral
except ImportError:
    pixtral = None


def _position_ids_in_meshgrid(*args, **kwargs):
    raise NotImplementedError("position_ids_in_meshgrid is not available in this transformers version")


if pixtral is not None:
    if not hasattr(pixtral, "PixtralRotaryEmbedding"):
        pixtral.PixtralRotaryEmbedding = pixtral.PixtralVisionRotaryEmbedding
    if not hasattr(pixtral, "position_ids_in_meshgrid"):
        pixtral.position_ids_in_meshgrid = _position_ids_in_meshgrid
