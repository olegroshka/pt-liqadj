from __future__ import annotations
import json
import re

from ptliq.web.site import default_example_payload_json


def test_default_example_payload_json_compact_and_minimal():
    s = default_example_payload_json()
    # Must be valid JSON
    obj = json.loads(s)
    assert isinstance(obj, dict) and "rows" in obj and isinstance(obj["rows"], list)
    rows = obj["rows"]
    assert len(rows) >= 2  # we expect at least two rows in the example
    # Rows must include only required fields
    for r in rows:
        assert set(r.keys()) == {"portfolio_id", "isin", "side", "size"}
    # No leftover feature keys from older examples
    assert "f_a" not in s and "f_b" not in s
    # Formatting: one JSON object per line inside rows (compact rows)
    # Count lines that start with a row object
    line_matches = re.findall(r"^\s*\{\"portfolio_id\":", s, flags=re.MULTILINE)
    assert len(line_matches) == len(rows)
