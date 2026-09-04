# Open Issues

This file records blockers found during reproduction. A strategy must be
skipped rather than changing the GranuleKV native protocol when it requires a
new descriptor format, MPS/completion semantics, a second transfer state
machine, or a different SSD-to-GPU path.

| Strategy | Missing capability | Trigger | GranuleKV backend issue | Minimal follow-up |
| --- | --- | --- | --- | --- |
| Quest | Real query-aware per-layer block scoring and attention consumer integration | Current tree only has a deterministic tail proxy; no Quest runtime is vendored | No | Keep the shared request API and add an external-source adapter that emits per-layer `SparseKVAccessPlan` selections plus consumer validation |
| HiSparse | Faithful SGLang selection/residency semantics are not yet wired into vLLM | Local SGLang code exists separately, but the current adapter only emits a proxy plan | No | Port only the policy output into `SparseKVAccessPlan`; do not copy the SGLang runtime |
| SolidAttention | Attention-inner speculative microtask and correction semantics | Existing API has layer-window requests, not an attention-kernel microtask stream | Not established | Skip until the policy can be expressed as ordinary staged requests; do not modify the native protocol |
