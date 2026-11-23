import torch
import torch.nn as nn


class ImportanceGuidedAdaptivePooling(nn.Module):
    """Pool variable-sized, importance-guided regions using channel-wise max.

    The default produces 56x56 output from a 112x112 feature map. Set
    ``output_size=None`` to halve each input spatial dimension, which is useful
    for the CIFAR-sized inputs used by this project.
    """

    def __init__(self, output_size=(56, 56), eps=1e-12):
        super().__init__()
        if output_size is not None:
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            if len(output_size) != 2 or min(output_size) < 1:
                raise ValueError("output_size must contain two positive integers")
            output_size = tuple(int(size) for size in output_size)
        self.output_size = output_size
        self.eps = float(eps)

    def _importance_map(self, x):
        # Boundaries are discrete, so importance is deliberately detached.
        features = x.detach().to(torch.float32)
        horizontal = (features[..., 1:] - features[..., :-1]).square().sum(dim=1)
        vertical = (features[..., 1:, :] - features[..., :-1, :]).square().sum(dim=1)

        importance = torch.zeros(
            (x.size(0), x.size(2), x.size(3)),
            device=x.device,
            dtype=torch.float32,
        )
        importance[..., :-1] += horizontal * 0.5
        importance[..., 1:] += horizontal * 0.5
        importance[..., :-1, :] += vertical * 0.5
        importance[..., 1:, :] += vertical * 0.5
        return importance + self.eps

    @staticmethod
    def _axis_boundaries(axis_importance, output_length):
        batch_size, input_length = axis_importance.shape
        if input_length < output_length:
            raise ValueError(
                f"Cannot make {output_length} non-empty regions from "
                f"{input_length} input positions"
            )
        if output_length == 1:
            return torch.tensor(
                [[0, input_length]], device=axis_importance.device, dtype=torch.long
            ).expand(batch_size, -1)

        # Prefix sums include both endpoints; quantiles select interior edges.
        prefix = torch.cat(
            (axis_importance.new_zeros((batch_size, 1)), axis_importance.cumsum(dim=1)),
            dim=1,
        )
        targets = prefix[:, -1:] * (
            torch.arange(1, output_length, device=axis_importance.device,
                        dtype=axis_importance.dtype).unsqueeze(0)
            / output_length
        )
        proposed = torch.searchsorted(prefix.contiguous(), targets.contiguous())

        # Enforce strictly increasing edges while leaving room for all later bins.
        edge_numbers = torch.arange(
            1, output_length, device=axis_importance.device
        ).unsqueeze(0)
        offsets = torch.cummax(proposed - edge_numbers, dim=1).values
        offsets = offsets.clamp(min=0, max=input_length - output_length)
        interior = offsets + edge_numbers
        return torch.cat(
            (
                torch.zeros((batch_size, 1), device=axis_importance.device,
                            dtype=torch.long),
                interior,
                torch.full((batch_size, 1), input_length,
                           device=axis_importance.device, dtype=torch.long),
            ),
            dim=1,
        )

    @staticmethod
    def _region_ids(boundaries, input_length):
        positions = torch.arange(input_length, device=boundaries.device)
        return torch.searchsorted(
            boundaries[:, 1:-1].contiguous(),
            positions.expand(boundaries.size(0), -1).contiguous(),
            right=True,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("input must have shape (batch, channels, height, width)")
        batch_size, channels, height, width = x.shape
        if self.output_size is None:
            output_height, output_width = height // 2, width // 2
        else:
            output_height, output_width = self.output_size
        if output_height < 1 or output_width < 1:
            raise ValueError("input spatial dimensions must be at least 2")
        if height < output_height or width < output_width:
            raise ValueError("each output region must contain an input position")

        importance = self._importance_map(x)
        height_importance = importance.sum(dim=2)
        width_importance = importance.sum(dim=1)
        height_boundaries = self._axis_boundaries(height_importance, output_height)
        width_boundaries = self._axis_boundaries(width_importance, output_width)

        row_ids = self._region_ids(height_boundaries, height)
        column_ids = self._region_ids(width_boundaries, width)
        region_ids = (
            row_ids.unsqueeze(2) * output_width + column_ids.unsqueeze(1)
        ).reshape(batch_size, 1, height * width)
        region_ids = region_ids.expand(-1, channels, -1)

        values = x.reshape(batch_size, channels, height * width)
        pooled = torch.full(
            (batch_size, channels, output_height * output_width),
            -torch.inf,
            dtype=x.dtype,
            device=x.device,
        )
        pooled = pooled.scatter_reduce(
            dim=2,
            index=region_ids,
            src=values,
            reduce="amax",
            include_self=True,
        )
        return pooled.reshape(batch_size, channels, output_height, output_width)


class LearnedWeightedMeanPooling(nn.Module):
    """Learn feature-dependent weights and average each non-overlapping 2x2 bin."""

    def __init__(self, channels):
        super().__init__()
        self.weight_predictor = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("input must have shape (batch, channels, height, width)")
        batch_size, channels, height, width = x.shape
        if height % 2 or width % 2:
            raise ValueError("weighted mean pooling requires even height and width")

        output_height, output_width = height // 2, width // 2
        scores = self.weight_predictor(x)
        scores = scores.reshape(batch_size, 1, output_height, 2, output_width, 2)
        scores = scores.permute(0, 1, 2, 4, 3, 5).reshape(
            batch_size, 1, output_height, output_width, 4
        )
        weights = torch.softmax(scores, dim=-1)

        regions = x.reshape(batch_size, channels, output_height, 2, output_width, 2)
        regions = regions.permute(0, 1, 2, 4, 3, 5).reshape(
            batch_size, channels, output_height, output_width, 4
        )
        return (regions * weights).sum(dim=-1)
