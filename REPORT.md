# REPORT — CAIM (From Modalities to Styles)

Lab twin of George & Marcel, *From Modalities to Styles* (IEEE TBIOM 2024 / arXiv:2404.14247). CelebA LR×4 stands in for HFR target modalities (see DEVIATIONS).

## Train

- CelebA train: n=20000, classes=5765
- Wall_s=235.18102383613586
- Protocol: full

## Metrics (ours)

| Protocol | Value |
|----------|------:|
| CelebA LR→HR Rank-1 | 4.05% |
| LFW accuracy | 57.12% |

Gallery IDs=300, probes=593, LFW pairs=6000.

## Figures

![Metrics](outputs/figures/full/metrics.svg)

## Regenerate

```bash
bash scripts/report.sh
```
