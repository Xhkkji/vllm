# LMCache Baseline + BaM Integration + Tutti Comparison Plan

## Goal

This plan defines the new mainline for experiments in `vllm-bam`.

We will no longer treat Mooncake as the primary development path.
Instead, we will:

1. Use LMCache as the baseline.
2. Integrate BaM underneath LMCache as a storage/backend tier.
3. Compare the resulting system against Tutti using aligned metrics.


## Main Decision

The core architectural decision is:

- Keep **LMCache** as the KV-cache runtime and request/chunk control plane.
- Use **BaM** as the lower-level storage/data movement backend.

This means:

- Do **not** start by building a brand-new standalone `BaMConnectorV1`.
- Do **not** continue the mainline around Mooncake.
- Do **not** keep BaM logic tied only to V0 `swap_in/swap_out`.

Instead:

- Preserve LMCache's scheduler/worker lifecycle.
- Reuse the existing BaM row/block machinery.
- Reorganize the BaM code into an LMCache-facing backend adapter.


## Why This Direction

### Why LMCache should be the baseline

LMCache already provides:

- a standard vLLM connector path,
- request-aware cache semantics,
- chunk-based KV lookup/store/retrieve,
- disaggregated prefill/decode examples,
- a cleaner baseline story for system comparison.

This makes LMCache a much better baseline than a custom Mooncake-centered path.

### Why BaM should live below LMCache

BaM is stronger as:

- a block/row storage engine,
- an SSD-backed KV backend,
- a data-plane optimization target.

BaM is weaker as:

- a full KV-cache runtime,
- a request-aware scheduler-facing cache service.

So the right division is:

- LMCache answers "what to cache, when to load, when to store".
- BaM answers "how to persist and fetch the backing KV payload efficiently".


## What Already Exists in `vllm-bam`

### Existing LMCache path

Relevant files:

- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
- `examples/lmcache/`
- `examples/lmcache/disagg_prefill_lmcache_v1/`

These already provide the baseline control flow:

- scheduler-side token matching,
- worker-side `start_load_kv`,
- worker-side `save_kv_layer`,
- worker-side `wait_for_save`.

### Existing BaM path

Relevant files:

- `vllm/worker/bam_row_store_loader.py`
- `vllm/worker/bam_block_store.py`
- `vllm/worker/bam_swap_reader.py`
- `vllm/worker/bam_shadow_writer.py`
- `vllm/worker/cache_engine.py`
- `vllm/envs.py`

These already provide:

- BaM row-store import and initialization,
- row/block packing logic,
- block restore logic,
- debug/verification hooks,
- experimental V0 swap integration.


## Reuse vs Reorganization

### Can be reused directly

These parts should be preserved with little or no change:

- `vllm/worker/bam_row_store_loader.py`
- BaM import path and environment variable handling in `vllm/envs.py`
- the general row-store usage model:
  - `store_rows(...)`
  - `load_rows(...)`

### Can be reused logically but should be moved/reframed

These files contain useful logic but are currently tied to V0 swap semantics:

- `vllm/worker/bam_block_store.py`
- `vllm/worker/bam_swap_reader.py`
- `vllm/worker/bam_shadow_writer.py`

What is useful in them:

- packing a KV block into BaM rows,
- reconstructing KV from row payloads,
- batching row operations,
- validation and bandwidth accounting.

What should not be preserved as-is:

- the dependence on `swap_in/swap_out`,
- the dependence on `cpu_block_id -> gpu_block_id` mapping,
- the assumption that BaM is only a V0 internal block swap path.

### Should not remain the main integration point

This path should no longer be the mainline integration target:

- `vllm/worker/cache_engine.py` BaM hooks for V0 swap.

We may still keep it for internal experiments, but it should not be the primary
BaM integration path going forward.


## Proposed Architecture

### High-level layering

1. **LMCacheConnectorV1**
   - unchanged outer interface
   - still owns scheduler/worker KV cache semantics

2. **LMCache-BaM adapter**
   - new layer between LMCache worker save/load and BaM row store
   - converts LMCache-oriented operations into BaM row operations

3. **BaM backend**
   - row-store initialization
   - row ID management
   - row payload load/store

### Intended data/control split

- **Control plane**: LMCache
  - request matching
  - cached token accounting
  - load/store decisions
  - chunk lifecycle

- **Data plane**: BaM
  - actual KV payload storage
  - SSD-backed row persistence
  - row readback and reconstruction


## New Code Organization

### New files to add

These should be added under the V1 connector area instead of the worker swap
path.

1. `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_bam_backend.py`
   - owns BaM row-store initialization
   - wraps row load/store primitives
   - exposes a stable backend API for LMCache-side use

2. `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_bam_layout.py`
   - maps LMCache chunk/span semantics to BaM row IDs
   - replaces the current V0-only `cpu_block_id -> row_offset` mindset

3. `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_bam_adapter.py`
   - worker-side adapter used by LMCache save/load hooks
   - translates between LMCache layer/chunk data and BaM backend calls

### Existing files likely to change

1. `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
   - minimal changes only
   - inject optional use of the LMCache-BaM adapter

2. `vllm/envs.py`
   - add LMCache+BaM specific flags if needed
   - keep current BaM import/cache/device knobs

### Existing files to leave alone initially

- `vllm/worker/cache_engine.py`
- V0 swap path code
- Mooncake connector/store files


## Integration Strategy

### Phase 1: LMCache baseline only

Goal:

- get a stable LMCache baseline,
- validate the disaggregated prefill/decode path,
- collect baseline metrics before any BaM changes.

Required outcome:

- reproducible logs,
- stable startup scripts,
- baseline TTFT/ITL/throughput numbers.

### Phase 2: BaM shadow store under LMCache

Goal:

- write KV payloads into BaM during LMCache save lifecycle,
- keep LMCache's original path as the source of truth.

Mode:

- `shadow`

Behavior:

- `save_kv_layer` writes to both LMCache normal path and BaM shadow backend,
- `start_load_kv` still loads through the original LMCache path.

Why:

- lowest risk,
- validates data extraction/packing,
- validates row-store correctness,
- creates a clean debug stage.

### Phase 3: Prefer-BaM load path

Goal:

- load from BaM first when enabled,
- fall back to the original LMCache path on miss or decode mismatch.

Mode:

- `prefer_bam`

Behavior:

- `start_load_kv` first attempts BaM-backed restore,
- if unavailable, fallback to standard LMCache retrieval.

Why:

- introduces real BaM data-plane benefit,
- keeps operational safety.

### Phase 4: BaM-primary mode

Goal:

- make BaM the primary payload backend for the LMCache path.

Mode:

- `only_bam`

Behavior:

- read/write path uses BaM as the primary external payload backend.

This should only happen after Phases 2 and 3 are stable.


## Recommended Runtime Flags

We should keep the current BaM environment knobs and add LMCache-specific BaM
mode knobs.

Suggested new knobs:

- `VLLM_LMCACHE_BAM_ENABLE=1`
- `VLLM_LMCACHE_BAM_MODE=shadow|prefer_bam|only_bam`
- `VLLM_LMCACHE_BAM_VERIFY=0|1`
- `VLLM_LMCACHE_BAM_LAYOUT=layer_block`

Keep using current BaM knobs where possible:

- `VLLM_BAM_IMPORT_PATH`
- `VLLM_BAM_CACHE_SIZE_MB`
- `VLLM_BAM_NUM_SSD`
- `VLLM_BAM_SSD_LIST`
- `VLLM_BAM_CTRL_IDX`


## Comparison Plan vs Tutti

We should compare at three levels.

### 1. System architecture comparison

Compare:

- LMCache baseline
- LMCache + BaM
- Tutti

Dimensions:

- external KV runtime design,
- cache unit (chunk/block),
- storage tiering,
- request awareness,
- async load/store behavior,
- deployment complexity.

### 2. End-to-end serving metrics

Primary metrics:

- TTFT
- ITL / decode latency
- end-to-end throughput
- external cache hit tokens / hit rate
- GPU memory pressure sensitivity

### 3. Backend/storage metrics

Primary metrics:

- KV save throughput
- KV load throughput
- per-request load latency
- large-chunk vs small-chunk behavior
- shadow vs prefer_bam vs only_bam


## Immediate Execution Plan

### Step 1

Run and stabilize the LMCache baseline in `vllm-bam`.

Deliverables:

- working launch commands,
- baseline logs,
- first metrics table.

### Step 2

Create the new BaM backend files under the V1 connector path.

Deliverables:

- backend skeleton,
- layout skeleton,
- adapter skeleton,
- no functional change yet.

### Step 3

Implement shadow-store only.

Deliverables:

- BaM writes during LMCache save path,
- correctness verification,
- shadow-store counters.

### Step 4

Enable optional prefer-BaM load path.

Deliverables:

- BaM-backed load in `start_load_kv`,
- fallback behavior,
- validation metrics.

### Step 5

Build a comparison matrix:

- LMCache baseline
- LMCache + BaM shadow
- LMCache + BaM prefer_bam
- Tutti-aligned reporting table


## Non-Goals for Now

We should explicitly avoid the following in the current phase:

- extending Mooncake further,
- building a standalone `BaMConnectorV1` first,
- rewriting LMCache core semantics,
- widening the old V0 swap path before the LMCache path is validated.


## Summary

The recommended mainline is:

- baseline with LMCache,
- reuse BaM low-level row/block machinery,
- move BaM out of the V0-only swap framing,
- reintroduce it as an LMCache backend/tier,
- compare the resulting system against Tutti using aligned system and backend
  metrics.
