"""Snapshot: frozen, on-disk market-data access."""

from datetime import date
from typing import Any


class Snapshot:
    """Frozen snapshot of market instruments at a reference date.

    Instruments are stored as dictionaries with keys:
    - isin: unique identifier
    - maturity: date
    - bid: bid price
    - ask: ask price
    - (and any other fields)

    All operations return new Snapshot instances (immutable).
    """

    def __init__(self, reference_date: date, instruments: list[dict[str, Any]]) -> None:
        """Initialize a Snapshot.

        Args:
            reference_date: The date this snapshot represents.
            instruments: List of instrument dictionaries.
        """
        self._reference_date = reference_date
        self._instruments = instruments

    @staticmethod
    def new(reference_date: date) -> "Snapshot":
        """Create a new empty snapshot.

        Args:
            reference_date: The date this snapshot represents.

        Returns:
            A new Snapshot with no instruments.
        """
        return Snapshot(reference_date, [])

    @property
    def reference_date(self) -> date:
        """Return the reference date of this snapshot."""
        return self._reference_date

    def __len__(self) -> int:
        """Return the number of instruments in this snapshot."""
        return len(self._instruments)

    def with_instrument(self, instr: dict[str, Any]) -> "Snapshot":
        """Return a new snapshot with the instrument added.

        Args:
            instr: Instrument dictionary.

        Returns:
            A new Snapshot with the instrument added.
        """
        new_instrs = [*self._instruments, instr]
        return Snapshot(self._reference_date, new_instrs)

    def instruments(self) -> list[dict[str, Any]]:
        """Return a copy of all instruments.

        Returns:
            A new list copy of instruments.
        """
        return list(self._instruments)

    def by_isin(self, isin: str) -> dict[str, Any]:
        """Return the instrument with the given ISIN.

        Args:
            isin: The ISIN to look up.

        Returns:
            The instrument dictionary.

        Raises:
            KeyError: If no instrument with that ISIN exists.
        """
        for instr in self._instruments:
            if instr.get("isin") == isin:
                return instr
        raise KeyError(isin)

    def has_isin(self, isin: str) -> bool:
        """Check if an instrument with the given ISIN exists.

        Args:
            isin: The ISIN to check.

        Returns:
            True if the ISIN exists, False otherwise.
        """
        return any(instr.get("isin") == isin for instr in self._instruments)

    def filter_by_maturity(self, start: date, end: date) -> list[dict[str, Any]]:
        """Filter instruments by maturity date range (inclusive).

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of instruments with maturity in [start, end].
        """
        result = []
        for instr in self._instruments:
            maturity = instr.get("maturity")
            if maturity is not None and start <= maturity <= end:
                result.append(instr)
        return result

    def time_to_maturity_years(self, instr: dict[str, Any]) -> float:
        """Calculate time to maturity in years using ACT/365F.

        Args:
            instr: Instrument dictionary with 'maturity' key.

        Returns:
            Time to maturity in years (ACT/365F).

        Raises:
            ValueError: If instrument has no maturity date.
            TypeError: If maturity is not a date.
        """
        maturity = instr.get("maturity")
        if maturity is None:
            raise ValueError("Instrument has no maturity date")
        if not isinstance(maturity, date):
            raise TypeError("maturity must be a date")

        days_diff = (maturity - self._reference_date).days
        return float(days_diff) / 365.0
