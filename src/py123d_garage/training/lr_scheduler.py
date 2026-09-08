"""Learning-rate schedule for training: cosine warm restarts with a per-cycle warmup ramp."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override


class CosineAnnealingWarmRestartsWithWarmup(LRScheduler):
    """
    Cosine annealing with warm restarts whose cycles open with a linear ramp.

    Cycle ``i`` runs for ``T_0 * T_mult**i`` steps: its first
    ``warmup_fraction`` of steps ramp linearly to the base learning rate and the
    rest anneal to ``eta_min``. A zero fraction is plain
    :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        warmup_fraction: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Build the schedule.

        Args:
            optimizer: Optimizer whose learning rates are set.
            T_0: Steps in the first cycle.
            T_mult: Factor each cycle's length grows by.
            warmup_fraction: Share of a cycle spent ramping up, in [0, 1).
            eta_min: Learning rate a cycle anneals down to.
            last_epoch: Step to resume from, or -1 to start fresh.

        Raises:
            ValueError: If the cycle length, growth factor or fraction is out of range.
        """
        if T_0 < 1 or T_mult < 1:
            raise ValueError(
                f"T_0 and T_mult must be at least 1, got {T_0}, {T_mult}",
            )
        if not 0.0 <= warmup_fraction < 1.0:
            raise ValueError(
                f"warmup_fraction must be in [0, 1), got {warmup_fraction}",
            )
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_fraction = warmup_fraction
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _cycle_position(self) -> tuple[int, int]:
        """
        Locate the current step within its cycle.

        Returns:
            The current cycle's length in steps and the step's index into it.
        """
        step = max(0, self.last_epoch)
        length = self.T_0
        while step >= length:
            step -= length
            length *= self.T_mult
        return length, step

    # Torch's stubs disagree on get_lr's return type across supported versions.
    @override
    def get_lr(self) -> list[float]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Inherited, see superclass."""
        length, step = self._cycle_position()
        warmup_steps = int(length * self.warmup_fraction)
        if step < warmup_steps:
            scale = (step + 1) / (warmup_steps + 1)
        else:
            progress = (step - warmup_steps) / max(1, length - warmup_steps)
            scale = (1.0 + math.cos(math.pi * progress)) / 2.0
        return [float(self.eta_min + (base - self.eta_min) * scale) for base in self.base_lrs]
