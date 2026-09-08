"""Aligning every recorded stream to one reference clock: the past grid, and the nearest-in-tolerance reads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypeVar, runtime_checkable

from py123d.datatypes import Timestamp

from py123d_garage.datatypes.numerics import NonNegativeInt


def past_offsets_us(horizon_us: int, interval_us: int | None) -> list[int]:
    """
    Past sampling offsets covering a horizon: one per interval, nearest first.

    Args:
        horizon_us: how far back the oldest sample reaches; 0 yields no offsets.
        interval_us: spacing between samples; required with a horizon.

    Returns:
        negative offsets relative to the anchor timestamp, e.g. [-100_000, -200_000].

    Raises:
        ValueError: if the horizon has no interval or is not a multiple of it.
    """
    if not horizon_us:
        return []
    if not interval_us:
        raise ValueError("a past horizon needs an interval: declare the sampling grid.")
    if horizon_us % interval_us:
        raise ValueError(
            f"horizon_us {horizon_us} is not a multiple of interval_us {interval_us}.",
        )
    return [-offset_us for offset_us in range(interval_us, horizon_us + interval_us, interval_us)]


@runtime_checkable
class TimestampedSample(Protocol):
    """A recorded sample that knows when it was recorded."""

    @property
    def timestamp(self) -> Timestamp:
        """The time the sample was recorded."""
        ...


SampleT = TypeVar("SampleT", bound=TimestampedSample)


def _sample_past_stream(
    fetch: Callable[[int, Literal["nearest", "backward"]], SampleT | None],
    anchor_timestamp_us: int,
    past_offsets_us: Sequence[int],
    tolerance_us: NonNegativeInt,
    stream_description: str,
) -> list[SampleT | None]:
    """
    Serves one recorded sample per past offset, or None where the stream had not started.

    A stream records on its own clock, so a target rarely lands on a recorded sample; the
    nearest one within the tolerance serves it. A target that misses while the stream was
    already recording is a dropout and raises, whereas a target before the stream's first
    sample is simply too early: near the start of a log or an episode there is nothing to
    serve, and the caller decides whether to skip it or repeat a neighbour.

    Args:
        fetch: returns the recorded sample matching a timestamp under a search criteria
        anchor_timestamp_us: recorded timestamp the offsets are measured from
        past_offsets_us: offsets to serve, one entry per returned sample
        tolerance_us: how far a recorded sample may sit from its target
        stream_description: names the stream in the dropout error

    Returns:
        one sample per offset, in the order given, None where the stream had not started

    Raises:
        ValueError: if a target misses while the stream was already recording
    """
    samples: list[SampleT | None] = []
    for offset_us in past_offsets_us:
        target_us = anchor_timestamp_us + offset_us
        sample = fetch(target_us, "nearest")
        if sample is not None and abs(sample.timestamp.time_us - target_us) <= tolerance_us:
            samples.append(sample)
            continue
        if fetch(target_us, "backward") is not None:
            raise ValueError(
                f"{stream_description} has no sample within {tolerance_us} µs of "
                f"{target_us} µs although the stream was already recording; it dropped out.",
            )
        samples.append(None)
    return samples
