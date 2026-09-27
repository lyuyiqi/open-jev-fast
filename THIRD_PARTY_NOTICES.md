# Third-party work used by open-jev-fast

The MIT license in `LICENSE` covers only the original code in this repository. Nothing below is relicensed; each project keeps its own license. This repository bundles **no** model weights, tokenizer files, datasets or third-party source code, apart from the small patch against causal-conv1d described below.

## What this repository builds on

| Project | License | How it is used here |
|---|---|---|
| [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) (Open-Jev contributors), commit `3308a15` | MIT | **The system being accelerated.** `src/server.py` imports Open-Jev at runtime and reuses its model loader (`jev.model.DecisionModel.load`), candidate-prompt construction (`jev.api.candidate_prompts`), serving stack (`jev.serving`) and HTTP server (`jev.server`). It wraps `DecisionModel.load` (to merge the LoRA adapter and build the fast backbone) and replaces `DecisionModel.forward`; the replacement follows Open-Jev's scoring convention (chat-templated prompt per candidate, last-token hidden state, scalar head, and the zero/score logit pair for yes/no questions). Temperature and probabilities are still computed by Open-Jev. The phase-1 scripts in `phase1/` patch Open-Jev in the same way. |
| [Open-Jev-27B-v1.1](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1) | Apache-2.0 | The LoRA adapter, decision head and temperature that are evaluated. Not redistributed. |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Alibaba Cloud), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Apache-2.0 | Base model and tokenizer required by Open-Jev-27B-v1.1. Not redistributed. |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) 5.10.2 | Apache-2.0 | The Qwen3.5 model implementation (`transformers.models.qwen3_5`) is the numerical reference. The CUDA kernels in `src/kernels.cu` are original code written to reproduce its bf16 rounding points (RMSNorm with `(1 + w)`, partial RoPE, gated attention output, Gated DeltaNet input path). `src/graph_patches.py` replaces two functions (`masking_utils._ignore_causal_mask_sdpa`, `Qwen3_5TextModel._update_linear_attn_mask`) at runtime with versions that always build the mask; no Transformers code is copied. `FastQwen35` reads the model's weights and its rotary-embedding module. |
| [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention) 0.5.2, Songlin Yang, Yu Zhang and contributors | MIT | On the prefix-tree path the Gated DeltaNet core is our own CUDA kernel (`gdn_fused*_k` in `src/kernels.cu`, original code, no FLA code copied). It implements the same chunked algorithm as FLA's `chunk_gated_delta_rule` (chunk size 64, WY representation, block-wise (I + A)⁻¹ in FLA's merge order) and keeps FLA's bf16/tf32 rounding points, so FLA is its numerical reference in `tests/test_r4_gdn.py`. `chunk_gated_delta_rule` is still called unchanged on the long-prefix path. Our kernels also reproduce the numerics of FLA's `l2norm_fwd` and `FusedRMSNormGated`. `bench/gdn_bench.py` also times `fused_recurrent_gated_delta_rule`. |
| [PEFT](https://github.com/huggingface/peft) 0.19.1 | Apache-2.0 | `merge_and_unload()` merges the LoRA adapter into the base weights. |
| [PyTorch](https://github.com/pytorch/pytorch) 2.14 | BSD-3-Clause | Runtime, `scaled_dot_product_attention` for full attention on the long-prefix path and for prefix trees above 384 rows (smaller trees use our tree-attention kernel), CUDA Graphs, `torch.compile` (phase 1), and `torch.utils.cpp_extension.load_inline` to build the kernels. |
| [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 1.7.0, Tri Dao | BSD-3-Clause | Used in phase 1 only. `patches/causal_conv1d-sm103.patch` changes its `setup.py` to build only for sm_103; the license is reproduced in `patches/LICENSE.causal-conv1d`. |
| NVIDIA cuBLAS / cuBLASLt (CUDA 13) | NVIDIA CUDA EULA | `src/lt.cpp` calls the public cuBLASLt API (heuristic algorithm query, algorithm enumeration and check, matmul). Linked at build time, not redistributed. |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek-AI) | MIT | Benchmarks only: `bench/dg_bench.py` and `bench/pdl_bench.py` time its sm100 bf16 GEMM against cuBLASLt. Not used by the server. |
| [KDA](https://github.com/NVlabs/kda) (NVIDIA) | CC-BY-4.0 (documentation), Apache-2.0 (code) | Its kernel-development workflow notes (contract, candidate list, profiling evidence) and KernelWiki guided round 4. No code or text is copied. |
| [JevBench](https://github.com/fstandhartinger/jevbench) (Florian Standhartinger and contributors), commit `f8ce713` | MIT (harness and original decisions; see its THIRD-PARTY.md for imported tasks) | Accuracy and latency evaluation. `bench/run_jevbench.py` calls its `TypeSafeAdapter` and `score_task` unchanged. `results/` stores only task IDs, predictions, correctness and latency, not task text. |
| [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (Eliahu Horwitz), adapted from the [Nerfies](https://nerfies.github.io) project page | CC BY-SA 4.0 | Layout of the report web page `docs/index.html` and its stylesheet `docs/static/css/index.css`. The page footer credits the template as it requests, and the page is under CC BY-SA 4.0. |
| [Bulma](https://bulma.io) 0.9.1 | MIT | CSS framework used by the web page (`docs/static/css/bulma.min.css`, unmodified). |
| TypeSafe typed-decision HTTP API ([docs](https://docs.typesafe.ai/api)) | — | The request format (`state`, `questions`, `noul`/`choice`/`score`) served by Open-Jev and therefore by this server. This project, like Open-Jev and JevBench, is not affiliated with or endorsed by TypeSafe AI. |

## Methods and related work

The techniques below are standard or published; our implementations are independent.

- **Gated DeltaNet** and its chunkwise-parallel form (Yang, Kautz and Hatamizadeh, ICLR 2025; Yang et al., NeurIPS 2024), as implemented in FLA; our CUDA kernel follows the same algorithm.
- **Online softmax over key blocks** (FlashAttention-2, Dao, ICLR 2024) in the tree-attention kernel, with the tree mask passed as bits.
- Inline PTX (`mma.sync`, `ldmatrix`, `cp.async`, `bar.sync`) as documented in NVIDIA's PTX ISA.
- **Split-K GEMM** (partial products over slices of K, then a reduction), a standard GEMM decomposition, e.g. in NVIDIA CUTLASS. Here the fp32 partial sums are reduced inside the fused residual + RMSNorm kernel.
- **Sharing a common prompt prefix across sequences**, as in Hydragen (Juravsky et al., 2024) and SGLang's RadixAttention (Zheng et al., NeurIPS 2024). Here it is done inside a single forward pass for scoring, without a KV cache.
- **Tree-structured attention masks over packed sequences**, as in SpecInfer (Miao et al., ASPLOS 2024) and Medusa (Cai et al., ICML 2024).
- **LoRA** (Hu et al., ICLR 2022), **RMSNorm** (Zhang and Sennrich, NeurIPS 2019), **RoPE** (Su et al., 2021).

Full references are in `README.md`.
