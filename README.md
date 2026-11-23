# ResNet_modified
The project is about implementing the ResNet architecture from scratch, and replicating the ReNet paper, and trying to modify it and compare the behaviour of the modified network with the original one

## ResNet variants

The project provides three pooling variants on the same CIFAR-style ResNet
backbone:

- `ResNet18`: classical 3×3 max pooling.
- `ImportanceGuidedResNet18`: partitions space from channel-aggregated squared
  feature differences and max-pools each variable-sized region.
- `LearnedWeightedResNet18`: predicts spatial weights from the features and
  computes a weighted mean in every 2×2 region. The predictor learns from the
  classification loss.

The importance pool's default output is 56×56 for a 112×112 feature map; in
the CIFAR model it halves 32×32 maps to 16×16. The weighted-mean pool also
halves each spatial dimension.

Run the two training configurations with the same seed to compare validation
accuracy and convergence:

```bash
python main.py --model classical --seed 42
python main.py --model importance --seed 42
python main.py --model learned --seed 42
```

Measure full-model inference latency, peak GPU memory, and profiler-reported
FLOPs for all three variants on the selected device:

```bash
python benchmark_pooling.py --input-size 112 --batch-size 1
```

Profiler FLOP totals depend on which operations the installed PyTorch version
supports; it may omit adaptive reduction operations. The benchmark reports
measured latency and memory separately and does not infer savings from region
sizes.

The pooling correctness checks are in `tests/test_importance_pool.py`.
