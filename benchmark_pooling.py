"""Measure full-model inference cost for all three pooling choices."""

import argparse
import time

import torch

from models.resnet18 import (
    ImportanceGuidedResNet18,
    LearnedWeightedResNet18,
    ResNet18,
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model_class, inputs, device, warmup, iterations):
    torch.manual_seed(42)
    model = model_class(num_classes=10).to(device).eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        synchronize(device)
        elapsed = time.perf_counter() - started
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else float("nan")
        )

        # FLOP coverage depends on PyTorch's profiler support for each operator.
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            model(inputs)

    total_flops = sum(event.flops for event in profile.key_averages())
    return elapsed * 1000 / iterations, peak_memory, total_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=112)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if args.batch_size < 1 or args.input_size < 2:
        parser.error("batch-size must be positive and input-size must be at least 2")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    inputs = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    print(f"Device: {device}; input: {tuple(inputs.shape)}")
    variants = (
        ("classical", ResNet18),
        ("importance", ImportanceGuidedResNet18),
        ("learned", LearnedWeightedResNet18),
    )
    for name, model_class in variants:
        latency, peak_memory, flops = benchmark(
            model_class, inputs, device, args.warmup, args.iterations
        )
        memory_text = f"{peak_memory:.2f} MiB" if device.type == "cuda" else "n/a (CPU)"
        print(
            f"{name}: {latency:.3f} ms/inference, peak GPU memory {memory_text}, "
            f"profiler-reported FLOPs {flops:,} (some reduction ops may be omitted)"
        )


if __name__ == "__main__":
    main()
