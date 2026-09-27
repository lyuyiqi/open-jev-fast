"""Remove CPU-sync shortcuts so the Qwen3.5 backbone forward can be CUDA-graph captured.
Both are pure optimizations in transformers (skip building a mask when there is no padding);
always building the mask gives the same math."""
import transformers.masking_utils as mu
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
mu._ignore_causal_mask_sdpa = lambda *a, **k: False
def _update_linear_attn_mask(self, attention_mask, past_key_values):
    return attention_mask
mq.Qwen3_5TextModel._update_linear_attn_mask = _update_linear_attn_mask
