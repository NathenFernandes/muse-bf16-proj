# Muse BF16 Projection Kernels

Nathen Fernandes's NVIDIA H200 solution for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/), problem `muse_bf16_proj`.

**Highest leaderboard rank achieved: #1. Best judged speedup: 1.5186× over SGLang.**

The #1 position was achieved during the competition with earlier submissions on October 24, 2026. It is a historical peak, not a claim of current first place or final competition victory. The later v18 submission is the best measured score in this repository. Leaderboard positions remain subject to the competition's final review.

## Results

| Version | Judge speedup | 95% interval | Job ID | Milestone |
|---|---:|---|---|---|
| v18 | 1.5186× | 1.4958–1.5413× | `66d8b007f6c641ee` | Best judged score; 9/9 tests passed |

The aggregate v18 result corresponds to about **34.15% lower latency** than the SGLang baseline under the competition's scoring method. Its cold preparation took **53.80 seconds**, with no judge flags. Scores come from actual judge runs, not projected sandbox performance.

| Load | SGLang total (µs) | Solution total (µs) | Speedup |
|---|---:|---:|---:|
| decode 1 | 3697.18 | 2194.80 | 1.6845× |
| decode 16 | 4724.90 | 2956.43 | 1.5982× |
| decode 32 | 4788.38 | 3394.59 | 1.4106× |
| verify 1 | 4718.94 | 2320.17 | 2.0339× |
| verify 16 | 5308.19 | 4149.79 | 1.2791× |
| verify 32 | 6312.10 | 5085.26 | 1.2413× |

Load totals include each operation's challenge weight. The overall score is the geometric mean of the six load speedups, not an arithmetic average of per-kernel ratios. The judge alternates candidate and SGLang CUDA-graph replays with cold L2 and subtracts fixed replay overhead. See [per-shape timings](results/v18_shapes.csv) and [judge result extracts](results/v18.json). The [historical ranking record](results/ranking-history.md) identifies the provenance of the #1 claim.

## Implementation

The timed operation is `out = x @ W.T`, covering attention QKV/gate and output projections, pooler, unpooler, and the FP32 router. Scored token counts are 1, 8, 16, 32, 128 and 256.

- Shape-specific CUDA kernels use Hopper WGMMA, TMA and cluster reductions where measured beneficial.
- Small-token paths use lossless BF16 weight packing with exact escape restoration and FP32 accumulation.
- Larger shapes selectively use pinned cuBLASLt algorithms.
- The router retains FP32 weights or the permitted BF16 hi/lo representation.
- Unsupported cases fall through the existing dispatch. Timed calls allocate fresh outputs and compute from the supplied activations.

`submission.py` is the exact v18 submission, unchanged from its judged bytes. CUDA source strings are compressed to meet the challenge's 1 MB submission limit. Use `python3 tools/expand_source.py submission.py /tmp/submission_expanded.py` to produce a readable inspection copy without importing or executing the submission.

## Running

Use a Linux NVIDIA H200 environment with CUDA supporting `sm_90a`, PyTorch, a C++ compiler, Ninja, and the challenge dependencies. The development image was `lmsysorg/sglang:dev-muse-spark-open`; that tag is mutable and is not a reproducibility guarantee.

Obtain the official harness separately with `okc init muse_bf16_proj`. In that initialized workspace, use this repository's `submission.py`. The interface is `prepare(weights)` once, followed by `kernel_fn(data, state)` for each input. See the official problem specification for tensor shapes and correctness bounds. No GPU is provisioned by this repository.

`okc check submission.py` runs official validation and consumes check quota. `okc submit submission.py` is a scored submission and consumes scored quota. Neither is run automatically.

## Validation and limitations

V18 passed all nine official judge tests. Additional local checks covered fresh-input graph replay, output retention, nearby shapes, lossless packing and sanitizer checks. Sandbox and H100 checks are distinct from the official H200 result.

An inherited limitation remains for extreme finite BF16 inputs outside the official generator: intermediate arithmetic can produce nonfinite outputs. This is not an all-finite-input correctness guarantee or a general-purpose GEMM replacement. Performance is specific to the measured H200 workload. The later v19 investigations did not produce a qualifying improvement and are not included as working optimizations.
