from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def py123d_garage_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_DATA_ROOT", "/data")
