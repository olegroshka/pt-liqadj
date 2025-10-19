import warnings

# Silence a third-party deprecation warning emitted by torch_geometric when importing llm backend utils.
# We do not import torch_geometric.distributed directly; this comes from torch_geometric internals.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"torch_geometric\.llm\.utils\.backend_utils",
)
