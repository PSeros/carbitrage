"""Guards on the acquisition modes."""

from __future__ import annotations

import pytest

from carbitrage import Lease
from carbitrage.errors import CarbitrageError


def test_an_open_calculation_lease_is_rejected_rather_than_approximated() -> None:
    with pytest.raises(CarbitrageError, match="residual risk on the lessor"):
        Lease(monthly_rate=199.0, residual_risk_borne_by="lessee")
