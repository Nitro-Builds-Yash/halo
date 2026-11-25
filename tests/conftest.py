import os
import tempfile

# Must run before `halo.core.config` is imported anywhere.
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp(prefix="halo_test_"))
os.environ["OPENAI_API_KEY"] = ""
os.environ["API_KEY"] = ""
