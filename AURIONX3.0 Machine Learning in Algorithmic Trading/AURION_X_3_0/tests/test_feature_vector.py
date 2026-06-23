from __future__ import annotations

from ml.feature_engineering import FEATURE_ORDER, FEATURE_VECTOR_LEN


def test_feature_vector_len_matches_order() -> None:
    assert FEATURE_VECTOR_LEN == len(FEATURE_ORDER) == 10
