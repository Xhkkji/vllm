# Runners

`plan_smoke` validates strategy composition without launching vLLM or touching
SSD. A future hardware runner should call the existing production connector
lifecycle and write a `MetricRecord`; it must not implement another lifecycle.
`run_formal.sh` is only a guarded pass-through to an explicitly supplied local
benchmark command.
