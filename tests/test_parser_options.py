import pytest

from issen_ccg.application.parser import ParseOptions


@pytest.mark.parametrize(
    "options",
    [
        {"arc_weight": -1},
        {"arc_weight": float("nan")},
        {"unary_cost": float("inf")},
        {"category_margin": -1},
        {"max_categories": 0},
        {"max_nodes": 0},
        {"timeout_ms": -1},
    ],
)
def test_invalid_search_policies_are_rejected(options):
    with pytest.raises(ValueError):
        ParseOptions(**options)
