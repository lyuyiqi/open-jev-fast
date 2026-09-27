"""jev.server with the optimized numerics (merged LoRA, fused RMSNorm, sync patches; causal-conv1d installed).
Used to check JevBench accuracy is unchanged by the optimizations."""
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import graph_patches  # noqa
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
def fused_forward(self, x):
    w = getattr(self, "_w1p", None)
    if w is None: w = self._w1p = (1.0 + self.weight.float()).contiguous()
    return F.rms_norm(x.float(), (x.shape[-1],), w, self.eps).type_as(x)
mq.Qwen3_5RMSNorm.forward = fused_forward
from jev import model as jm
_orig = jm.DecisionModel.load.__func__
def _load(cls, *a, **k):
    m = _orig(cls, *a, **k)
    if hasattr(m.backbone, "merge_and_unload"):
        m.backbone = m.backbone.merge_and_unload()
    print("OPT: LoRA merged, fused RMSNorm, sync patches; causal_conv1d =", mq.causal_conv1d_fn is not None, flush=True)
    return m
jm.DecisionModel.load = classmethod(_load)
from jev.server import main
sys.argv = ["jev.server"] + sys.argv[1:]
main()
