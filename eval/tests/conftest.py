"""
Test configuration for the eval suite.

Puts ``eval/`` on the import path so the tests can import ``grading`` and
``run_eval`` directly. Nothing here touches Chroma, Ollama or the network.
"""

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

GOLDEN_PATH = EVAL_DIR / "golden.json"


@pytest.fixture(scope="session")
def golden():
    """The shipped golden set, as loaded from disk."""
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="session")
def entries(golden):
    """Golden questions keyed by id, e.g. ``entries["q07"]``."""
    return {question["id"]: question for question in golden["questions"]}
