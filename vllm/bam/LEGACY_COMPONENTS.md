# Legacy BaM MDS components

The files under `vllm/bam/mds/` and the compatibility module
`vllm/bam/mds_connector.py` are retained for historical experiments and
rollback only. They are not the canonical GranuleKV transport boundary.

The active split is:

- `vllm/granulekv/`: vLLM-side layout and scheduling transport adapter.
- `BaM_IOStack/gids_module/granulekv/`: I/O client, service, protocol and
  memory ownership.
- `vllm/core/`: scheduler and block-reservation policy.

Do not add new active imports from `vllm.bam.mds` or `bam_mds`.
