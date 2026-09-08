"""Cache-source bookkeeping for routes the evaluation side provides at runtime."""

from __future__ import annotations

from py123d.api.utils.cache_source_utils import utc_now_iso
from py123d.datatypes import CacheSourceInfo

PROVIDED_ROUTE_PRODUCER = "py123d_garage:provided_route@1"


def provided_route_source_info(external_source: str) -> CacheSourceInfo:
    """
    The ``CacheSourceInfo`` of a route that comes from outside a log.

    Args:
        external_source: where the route came from, e.g. the simulator that handed it over

    Returns:
        source info without log modalities to re-hash
    """
    return CacheSourceInfo(
        computed_by=PROVIDED_ROUTE_PRODUCER,
        computed_at=utc_now_iso(),
        source_modalities=[],
        external_sources=[external_source],
    )
