"""Shared fixtures."""

from __future__ import annotations

import pytest
from _tiny_bundle import MAPPING, make_tiny_bundle


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory) -> dict:
    """A real (random-init, 64 px) bundle exported through the release code path."""
    if not MAPPING.is_file():
        pytest.skip("tracked class mapping missing")
    return make_tiny_bundle(tmp_path_factory.mktemp("tiny_bundle"))
