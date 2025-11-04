from __future__ import annotations
import warnings

# Silence deprecations emitted by external libraries during tests.
# We keep our code clean; these warnings come from FastAPI/Gradio internals.
warnings.filterwarnings(
    "ignore",
    message=r".*Please use `import python_multipart` instead.*",
    category=PendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*on_event is deprecated.*",
    category=DeprecationWarning,
)
# As a last resort in CI, ignore all Deprecation and PendingDeprecation warnings
# from third-party libs so test output is clean.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
