# Metrics Contract

Each smoke or formal run writes one JSON summary using `MetricRecord`. The
same record can be converted to CSV with:

```bash
python -m evaluation.paper_reproduction.common.metrics.summarize \
  evaluation/paper_reproduction/quest/results/formal \
  evaluation/paper_reproduction/quest/results.csv
```

Missing measurements remain null. A run must not fabricate overlap or
correctness values when the corresponding marker was not observed.
