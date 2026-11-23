import torch

from models.importance_pool import ImportanceGuidedAdaptivePooling
from models.importance_pool import LearnedWeightedMeanPooling
from models.resnet18 import (
    ImportanceGuidedResNet18,
    LearnedWeightedResNet18,
    ResNet18,
)


def test_output_shape_for_112_input():
    pool = ImportanceGuidedAdaptivePooling()
    output = pool(torch.randn(2, 5, 112, 112))
    assert output.shape == (2, 5, 56, 56)


def test_partitions_cover_every_position_and_are_nonempty():
    pool = ImportanceGuidedAdaptivePooling()
    features = torch.randn(2, 3, 112, 112)
    importance = pool._importance_map(features)
    height_edges = pool._axis_boundaries(importance.sum(dim=2), 56)
    width_edges = pool._axis_boundaries(importance.sum(dim=1), 56)
    rows = pool._region_ids(height_edges, 112)
    columns = pool._region_ids(width_edges, 112)
    region_ids = (rows.unsqueeze(2) * 56 + columns.unsqueeze(1)).reshape(2, -1)

    for sample_regions in region_ids:
        counts = torch.bincount(sample_regions, minlength=56 * 56)
        assert counts.sum().item() == 112 * 112
        assert torch.all(counts > 0)


def test_gradients_flow_through_pooled_values():
    pool = ImportanceGuidedAdaptivePooling()
    features = torch.randn(1, 2, 112, 112, requires_grad=True)
    pool(features).sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad).item() > 0


def test_uniform_importance_gives_two_by_two_regions():
    pool = ImportanceGuidedAdaptivePooling()
    features = torch.ones(1, 1, 112, 112)
    importance = pool._importance_map(features)
    height_edges = pool._axis_boundaries(importance.sum(dim=2), 56)
    width_edges = pool._axis_boundaries(importance.sum(dim=1), 56)
    assert torch.all(torch.diff(height_edges[0]) == 2)
    assert torch.all(torch.diff(width_edges[0]) == 2)


def test_high_importance_rows_and_columns_get_smaller_intervals():
    pool = ImportanceGuidedAdaptivePooling()
    features = torch.zeros(1, 1, 112, 112)
    features[:, :, 56:, :] = 1
    features[:, :, :, 56:] += 1
    importance = pool._importance_map(features)
    height_edges = pool._axis_boundaries(importance.sum(dim=2), 56)
    width_edges = pool._axis_boundaries(importance.sum(dim=1), 56)
    assert torch.diff(height_edges).min().item() == 1
    assert torch.diff(width_edges).min().item() == 1


def test_learned_weighted_mean_pooling_propagates_weight_gradients():
    pool = LearnedWeightedMeanPooling(channels=3)
    features = torch.randn(2, 3, 8, 8, requires_grad=True)
    pool(features).square().mean().backward()
    assert features.grad is not None
    assert pool.weight_predictor.weight.grad is not None
    assert torch.count_nonzero(pool.weight_predictor.weight.grad).item() > 0


def test_three_resnet_variants_share_classification_interface():
    features = torch.randn(2, 3, 32, 32)
    for model_class in (
        ResNet18,
        ImportanceGuidedResNet18,
        LearnedWeightedResNet18,
    ):
        assert model_class()(features).shape == (2, 10)
