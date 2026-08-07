"""Oracle price reconstruction and data-quality gate."""

from __future__ import annotations


class OracleGate:
    """Gates data quality by requiring a sustained streak of oracle-price agreement.

    The gate opens only after ``required_streak`` consecutive ticks whose
    reconstructed mid-price lies within ``threshold_bps`` basis points of the
    oracle mid-price.  Any tick that exceeds the threshold closes the gate and
    resets the streak.  A timestamp gap larger than ``gap_seconds`` between two
    consecutive calls to :meth:`update` also resets the streak.

    Parameters
    ----------
    market:
        Coin symbol (e.g. ``'BTC'``).
    threshold_bps:
        Maximum allowed deviation in basis points for a tick to count toward
        the streak.
    required_streak:
        Number of consecutive conforming ticks required to open the gate.
    gap_seconds:
        Maximum allowed time gap (seconds) between consecutive ticks before
        the streak resets.
    """

    def __init__(
        self,
        market: str,
        threshold_bps: float = 25.0,
        required_streak: int = 12,
        gap_seconds: float = 30.0,
    ) -> None:
        self.market = market
        self.threshold_bps = threshold_bps
        self.required_streak = required_streak
        self.gap_seconds = gap_seconds

        self._streak: int = 0
        self._open: bool = False
        self._last_ts: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: float,
        reconstructed_midpoint: float,
        oracle_midpoint: float,
    ) -> bool:
        """Process one tick and update gate state.

        Parameters
        ----------
        timestamp:
            Unix timestamp of the tick (seconds).
        reconstructed_midpoint:
            LOB-reconstructed mid-price for this tick.
        oracle_midpoint:
            Official oracle mid-price for this tick.

        Returns
        -------
        ``True`` if the gate is open after processing this tick.
        """
        # Time-gap check: reset streak (but not _last_ts yet) when gap is too large
        if self._last_ts is not None and (timestamp - self._last_ts) > self.gap_seconds:
            self.reset()

        self._last_ts = timestamp

        if self._within_threshold(reconstructed_midpoint, oracle_midpoint):
            self._streak += 1
            if self._streak >= self.required_streak:
                self._open = True
        else:
            # Out-of-threshold tick closes the gate and resets the streak
            self._streak = 0
            self._open = False

        return self._open

    def is_open(self) -> bool:
        """Return ``True`` if the gate is currently open."""
        return self._open

    def reset(self) -> None:
        """Force-close the gate and reset streak; preserve last timestamp."""
        self._streak = 0
        self._open = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _within_threshold(self, recon: float, oracle: float) -> bool:
        """Return ``True`` if ``|recon - oracle| / oracle * 10 000 <= threshold_bps``."""
        return abs(recon - oracle) / oracle * 10_000.0 <= self.threshold_bps
