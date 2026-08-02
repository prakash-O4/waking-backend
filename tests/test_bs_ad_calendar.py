from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.authority.bs_ad_calendar import BeyondCalendarRange, lookup


def test_epoch() -> None:
    assert lookup(2000, 1, 1) == (date(1943, 4, 14), False)


def test_2080_baisakh_starts_in_april_2023() -> None:
    ad_date, warning = lookup(2080, 1, 1)
    assert date(2023, 4, 12) <= ad_date <= date(2023, 4, 16)
    assert warning is False


def test_out_of_range_year_raises() -> None:
    with pytest.raises(BeyondCalendarRange):
        lookup(9999, 1, 1)


def test_invalid_day_raises() -> None:
    with pytest.raises(BeyondCalendarRange):
        lookup(2000, 1, 0)


def test_day_offset_monotonicity() -> None:
    first, _ = lookup(2050, 3, 1)
    second, _ = lookup(2050, 3, 2)
    assert second == first + timedelta(days=1)
