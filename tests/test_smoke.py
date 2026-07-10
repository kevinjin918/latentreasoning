"""Mock-first smoke test: import the package, no GPU/weights/data."""

import latentreasoning


def test_version():
    assert latentreasoning.__version__ == "0.0.1"
