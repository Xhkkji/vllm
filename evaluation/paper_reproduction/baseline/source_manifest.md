# Source Manifest

Record the vLLM commit, GranuleKV commit, model revision, SSD image identity,
HBM limit, workload, and environment variables for each baseline run.

Validated transport runs on 2026-09-03:

| Run | Payload | Result |
| --- | --- | --- |
| `20260903_granulekv_transport_smoke` | 8 blocks, 4 compute repeats | sync/async marker and payload verification passed; async overlap observed |
| `20260903_granulekv_transport_formal` | 128 blocks, 80 compute repeats | sync/async marker and payload verification passed; async overlap observed, but this configuration was `0.767x` end-to-end versus sync |

These are transport-baseline measurements, not faithful Quest, HiSparse,
SolidAttention, or Bidaw reproduction results.
