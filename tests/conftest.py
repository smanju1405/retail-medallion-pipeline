"""Test setup: point every output folder at a throwaway directory before core.config is imported."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="medallion-tests-")
os.environ.setdefault("GROQ_API_KEY", "")  # tests never call the network
