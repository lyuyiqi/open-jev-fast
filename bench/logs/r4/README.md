# Round-4 logs

Recorded output of the round-4 experiments (job 124522, one NVIDIA B300, 2026-09-27).

| File | Contents |
|---|---|
| `round4_benchmark.csv` | Every candidate change with its end-to-end latency, numerical check and notes, kept or rejected |
| `gdn_kernel.log` | Gated DeltaNet kernel versions vs FLA: accuracy, GPU time, time vs path length, per-phase times |
| `tree_attention.log` | Tree attention kernel vs cuDNN SDPA: accuracy, GPU time, A/B end to end |
| `prof_ex1_v3.log`, `prof_ex5_final.log` | Per-kernel GPU time for the example request before and after round 4 (`bench/prof_example.py`) |
| `lt_search_M288.log` | Exhaustive cuBLASLt search for the model's GEMM shapes at 288 rows (`bench/lt_search.py`) |
| `dg_vs_cublas_M288.log` | DeepGEMM vs cuBLASLt for the same shapes (`bench/dg_bench.py`) |
| `sk_e2e.log`, `v6_e2e.log` | End-to-end comparisons for split-K settings and the Gated DeltaNet kernel versions |
