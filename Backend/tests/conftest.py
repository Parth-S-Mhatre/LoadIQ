import os
import tempfile
from pathlib import Path

# The documents store is created at import time; point it at a throwaway
# directory before any test imports the app.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="loadiq-documents-tests-"))
os.environ.setdefault("DOCUMENTS_DB_PATH", str(_TEST_ROOT / "documents.db"))
os.environ.setdefault("DOCUMENTS_STORAGE_DIR", str(_TEST_ROOT / "uploads"))
os.environ.setdefault("DOCUMENTS_AUTH_MODE", "none")
