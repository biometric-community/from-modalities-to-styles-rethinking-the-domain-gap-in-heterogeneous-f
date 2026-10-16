# Deviations

| ID | Topic | Paper | Ours |
|----|-------|-------|------|
| D1 | Datasets | Tufts, MCXFace, Polathermal, SCFace, CUFSF, CASIA NIR-VIS | CelebA HR/LR×4 + LFW |
| D2 | Backbone | InsightFace IResNet100 (MS1M) | ImageNet ResNet18 with frozen stem+layer1 |
| D3 | CAIM placement | Multiple stages in IResNet | CAIM after layer2/3/4 |
| D4 | Epochs | 50, batch 90 | 5 epochs, batch 64, max 20k images |
| D5 | Upstream | Idiap bob.paper CAIM | Reimplemented from paper/arXiv description |
