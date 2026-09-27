# open-jev-fast

**Report:** [web page](https://lyuyiqi.github.io/open-jev-fast/) · [PDF](docs/report.pdf)

**A faster inference backend for [Open-Jev](https://github.com/Zefan-Cai/Open-Jev).** It serves the same HTTP API as Open-Jev's `jev.server`, with the same model ([Open-Jev-27B-v1.1](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1) on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), in bf16 with no quantization. It replaces the model forward pass with hand-written fused CUDA kernels, a prefix tree that computes shared prompt text once, tuned matrix multiplies and CUDA Graphs.

Open-Jev is by the Open-Jev contributors; this repository runs on top of it and needs an Open-Jev installation and the model weights. **Built on:** Open-Jev [1], Open-Jev-27B-v1.1 / Qwen3.8-27B [2, 3], flash-linear-attention [4], Hugging Face Transformers [7], PEFT [8], PyTorch [9]; evaluated with JevBench [11]. What is used from each project is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Results (1× NVIDIA B300, bf16)

| | Original Open-Jev | open-jev-fast | Speedup |
|---|---|---|---|
| Example request, one inference (3 questions, 7 candidates, 539 tokens; no HTTP) | 109.9 ms | **20.1 ms** | 5.5× |
| JevBench, mean per-task latency (231 tasks, HTTP, concurrency 1) | 703 ms | **51.3 ms** | 13.7× |
| JevBench P50 / P95 / max | 171 / 1272 / 16018 ms | **24.2 / 148 / 583 ms** | |
| JevBench accuracy (our runs) | 198/231 | 197/231 | one coin-flip task changed, see below |

![Latency ladder](docs/ladder.png)

The Open-Jev-27B-v1.1 model card reports 197/231 on public JevBench for this checkpoint; our local run of the original server scored 198/231.

## Test conditions

**Single inference (example request).**
- Request: Open-Jev's `configs/example-request.json`, a customer-service conversation with 3 questions (routing, 1 of 3; refund review, yes/no; urgency, 3 levels). That is 7 candidate prompts and 539 tokens.
- Timed: one full forward pass plus the scoring head over all 64 layers. Nothing is reused across requests; shared prefixes are computed once only within a request. Excludes tokenization (about 1 ms) and HTTP.
- Statistic: 5 warm-up runs, then the median over repeated runs, with `torch.cuda.synchronize()` before and after each. 20.1 ms is the median of 20 CUDA Graph replays (`tests/test_gtree.py`); every other step, including the original's 109.9 ms, is the median of 30 runs.

**JevBench.**
- The 231 public tasks at upstream `f8ce713`, run once in order at concurrency 1.
- Latency is client-side, as recorded by JevBench's own `TypeSafeAdapter`: tokenization, HTTP and JSON included, with a new connection per task (about 25 ms more than keep-alive).
- Optimized numbers are from a second pass, after the CUDA Graphs for the needed shapes were captured. Scoring uses JevBench's `score_task`.
- Both the original `jev.server` and this server ran with `--max-length 16384`, so that every JevBench task fits. The model card's saved limit is 4,096 tokens per candidate.
- Server-side inference alone has a P50 of 18.6 ms.

Per-task results for every version are in [`results/`](results/) (task ID, prediction, correctness, latency); [`results/summary.json`](results/summary.json) has the aggregates.

## How it works

The original is **CPU-bound**: each request launches about 5000 small GPU kernels, and the GPU is busy for only about 51 of the 106 ms. The seven candidate prompts also repeat most of their text.

**Phase 1, PyTorch level (110 → 32.4 ms).**
- Merge the LoRA adapter into the base weights.
- Fuse RMSNorm.
- Remove two CPU synchronizations in the Transformers mask code (`src/graph_patches.py`; both are "skip the mask if there is no padding" shortcuts, so the math is unchanged).
- Capture the whole model as one CUDA Graph at the exact length.
- Use the causal-conv1d kernel [6] and `torch.compile`.

**Phase 2, hand-written CUDA (32.4 → 20.1 ms).** This path replaces the phase-1 path.
1. **Six fused kernels** (`src/kernels.cu`) take over every non-matmul operation in a layer:
   - residual add + RMSNorm
   - SiLU·mul
   - Gated DeltaNet input prep: causal conv, SiLU, q/k/v split, head expansion, L2 norm, gate parameters
   - gated RMSNorm
   - full-attention prep: QK norm, partial RoPE, layout
   - sigmoid-gate multiply

   Matrix multiplies are merged from 9 to 4 per layer, and kernels per request drop from about 5000 to 945. Each kernel reproduces the bf16 rounding points of the Transformers reference [7] and FLA [4]. The Gated DeltaNet core itself is FLA's `chunk_gated_delta_rule` [4, 5], unchanged, and full attention is PyTorch SDPA.
2. **Prefix tree** (`src/fastmodel.py`: `gtree_build`, `forward_gtree`). One packed forward pass computes the shared context once, each question's text once, and each candidate's tail, so the weights are read once. For the example request, computed rows go from 574 to 277.
   - Linear-attention layers run each candidate over its complete path in a replicated layout, so the convolution window and recurrent state match the original.
   - Full-attention layers use an ancestor mask.
   - Long single-question prompts hand the prefix's final recurrent state to each candidate instead.

   Related ideas: shared-prefix attention [12, 13] and tree attention masks [14, 15].
3. **Matrix multiplies** (`src/lt.cpp`):
   - split-K with fp32 partial sums, reduced inside the RMSNorm kernel
   - every cuBLASLt heuristic algorithm timed per exact shape, keeping the fastest
   - the replication copies removed
4. **Serving** (`src/server.py`):
   - CUDA Graphs cached per (row bucket, candidates, path-length bucket), up to 512
   - buffered access log
   - `TCP_NODELAY`

A step-by-step breakdown, the GPU time budget and the failed attempts are in the report.

## Correctness

We checked at three levels:
1. **Kernels** are compared element by element with the PyTorch/FLA reference on real activations: 98–100% bit-identical ([`bench/logs/correct.log`](bench/logs/correct.log)).
2. **Hidden states** are compared layer by layer across all 64 layers. The last-token hidden state read by the head differs by 0.19% relative.
3. **JevBench** runs end to end, with predictions compared task by task.

Probability differences from the original grow with input length, because bf16 rounding-order differences accumulate through 64 layers and the recurrent state:
- **Example request:** at most **0.0026**.
- **Longest request tested** (10,722 tokens): up to **0.035**, with the decision unchanged ([`bench/logs/gtree2.log`](bench/logs/gtree2.log)).

On JevBench, two tasks changed prediction versus the original:
- `hard-opus-c-long_policy-03` is wrong both before and after.
- `hard-opus-a-temporal_numeric-09` is a coin flip: the original gives no/yes = 0.504/0.496, merging LoRA alone gives 0.503/0.497, and the hand-written kernels give 0.496/0.504.

## Scope and limitations

- Specialized for **one model** (Open-Jev-27B-v1.1 / Qwen3.8-27B shapes).
- **Tested only on an NVIDIA B300.** The kernels are plain CUDA C++ compiled for `sm_103` by default. For another GPU with enough memory, change the `-gencode` flag in `src/ext.py`. That path is untested, and the speedups will differ.
- bf16 only. FP8 GEMMs were measured (matrix-multiply time 14.6 → 7.8 ms) but are not used, since they change the numerics.
- The server processes requests one at a time.
- **GPU memory: about 96 GiB** (98,280 MiB in `nvidia-smi` after the full JevBench run), so an 80 GB GPU is not enough as is.
  - This covers the weights, the merged and split weight copies, the CUDA Graph pool and activations.
  - The original weights are kept next to the merged copies, so a large part is duplicated. Freeing them is not implemented.
  - The tests in `tests/` need about as much.
- The first request of a new shape captures a graph, adding 1–3 s.

## Quick start

Requirements: a GPU with about 96 GiB of free memory (tested only on an NVIDIA B300), CUDA 13 (`nvcc`, cuBLASLt), and Python with Open-Jev installed (commit `3308a15`) and its dependencies. We used torch 2.14 (cu130), transformers 5.10.2, peft 0.19.1, flash-linear-attention 0.5.2 and triton 3.8.0.

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git && (cd Open-Jev && git checkout 3308a15 && pip install -e '.[train]')
pip install flash-linear-attention==0.5.2
hf download ZefanCai/Open-Jev-27B-v1.1 --local-dir ./open-jev-27b-v1.1   # Open-Jev's loader fetches the pinned Qwen3.8-27B base
export OPEN_JEV_DIR=$PWD/Open-Jev OJ_CKPT=$PWD/open-jev-27b-v1.1/package/checkpoint
export CUDA_HOME=/path/to/cuda-13   # needs bin/nvcc, include/, lib/libcublasLt.so

./scripts/launch_server.sh          # same API as jev.server: POST http://localhost:18791/v1/systemone
```

Kernels are compiled on first import (`torch.utils.cpp_extension.load_inline`).

Correctness and benchmarks (each loads the original model as the reference):

```bash
export JEVBENCH_DIR=/path/to/jevbench   # https://github.com/fstandhartinger/jevbench @ f8ce713 (long test inputs + runner)
python tests/test_correct.py        # kernel-by-kernel, 64-layer and probability comparison
python tests/test_gtree.py          # two-level prefix tree vs original + CUDA Graph timing (the 20.1 ms number)
python bench/run_jevbench.py http://localhost:18791 open-jev out.json
```

## Repository layout

| Path | Contents |
|---|---|
| `src/` | CUDA kernels (`kernels.cu`, built by `ext.py`), cuBLASLt wrapper (`lt.cpp`, `lt_ext.py`), fast backbone and prefix tree (`fastmodel.py`), Transformers patches (`graph_patches.py`), server (`server.py`) |
| `tests/` | Correctness tests against the original implementation |
| `bench/` | Benchmarks (GEMM, split-K, cuBLASLt, FLA, profiling) and the JevBench runner; `bench/logs/` has their recorded output |
| `phase1/` | Phase-1 PyTorch-level scripts (LoRA merge, sync removal, CUDA Graph, torch.compile) |
| `patches/` | causal-conv1d build patch for sm_103 (phase 1 only) and its license |
| `results/` | JevBench per-task results for the original and each optimized version |
| `docs/` | Report web page (`index.html`, served by GitHub Pages), PDF report, figures, and the scripts that generate them |

## References

1. Open-Jev contributors. *Open-Jev.* https://github.com/Zefan-Cai/Open-Jev (commit `3308a15`), 2026. MIT.
2. *Open-Jev-27B-v1.1.* https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1. Apache-2.0.
3. Qwen Team, Alibaba Cloud. *Qwen3.8-27B.* https://huggingface.co/Qwen/Qwen3.8-27B (revision `1d4bf0f`). Apache-2.0.
4. Songlin Yang and Yu Zhang. *FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism.* https://github.com/fla-org/flash-linear-attention, 2024.
5. Songlin Yang, Jan Kautz and Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. https://arxiv.org/abs/2412.06464 See also Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen and Yoon Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. https://arxiv.org/abs/2406.06484
6. Tri Dao. *causal-conv1d.* https://github.com/Dao-AILab/causal-conv1d. BSD-3-Clause.
7. Thomas Wolf et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020 System Demonstrations. https://aclanthology.org/2020.emnlp-demos.6
8. Sourab Mangrulkar et al. *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.* https://github.com/huggingface/peft, 2022.
9. Jason Ansel et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.* ASPLOS 2024.
10. Edward J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
11. Florian Standhartinger and contributors. *JevBench.* https://github.com/fstandhartinger/jevbench (commit `f8ce713`), 2026. MIT.
12. Jordan Juravsky et al. *Hydragen: High-Throughput LLM Inference with Shared Prefixes.* 2024. https://arxiv.org/abs/2402.05099
13. Lianmin Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. https://arxiv.org/abs/2312.07104
14. Xupeng Miao et al. *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. https://arxiv.org/abs/2305.09781
15. Tianle Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. https://arxiv.org/abs/2401.10774
16. Biao Zhang and Rico Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019. https://arxiv.org/abs/1910.07467
17. Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. https://arxiv.org/abs/2104.09864
18. NVIDIA. *cuBLAS / cuBLASLt documentation.* https://docs.nvidia.com/cuda/cublas/ ; *CUTLASS* (split-K GEMM). https://github.com/NVIDIA/cutlass.

## License

MIT for the code in this repository ([LICENSE](LICENSE)). Models, datasets and third-party libraries keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
