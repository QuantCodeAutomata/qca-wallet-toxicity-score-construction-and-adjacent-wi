"""LOB reconstruction engine with zero-corruption policy and oracle gating.

Block envelopes are dicts with the following schema::

    {
        "block_number": int,
        "timestamp":    float,          # Unix seconds
        "market":       str,
        "updates":      [               # list of level diffs
            {"side": "bid"|"ask", "price": float, "qty": float},
            ...
        ],
        "oracle_price": float | None,   # optional; gates emission when present
    }

The :class:`ReplayEngine` applies the zero-corruption policy:

* **Block-number rewind** – if ``block_number`` ≤ last seen block for that
  market, the book and oracle gate are reset and ``None`` is returned.
* **Crossed book** – after applying all level diffs, if ``best_bid ≥ best_ask``
  the book and gate are reset and ``None`` is returned.
* **Negative depth** – if any update carries a negative qty the book and gate
  are reset and ``None`` is returned.

After any reset the oracle gate must reopen (streak of 12 in-threshold ticks)
before block-close states are emitted again.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .oracle_gate import OracleGate

logger = logging.getLogger(__name__)


class LimitOrderBook:
    """Single-market limit order book with dictionary-backed price levels.

    Bids and asks are stored as ``{price: qty}`` dicts.  Levels with ``qty=0``
    are removed on update.
    """

    def __init__(self) -> None:
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, side: str, price: float, qty: float) -> None:
        """Apply a single level diff.

        Parameters
        ----------
        side:
            ``'bid'`` or ``'ask'``.
        price:
            Price level to update.
        qty:
            New quantity at this level.  ``0.0`` removes the level.

        Raises
        ------
        ValueError
            If ``qty`` is negative (triggers zero-corruption policy upstream).
        """
        if qty < 0.0:
            raise ValueError(
                f"Negative qty {qty:.6f} for {side} @ {price:.6f}"
            )
        book = self.bids if side == "bid" else self.asks
        if qty == 0.0:
            book.pop(price, None)
        else:
            book[price] = qty

    def reset(self) -> None:
        """Clear all price levels on both sides."""
        self.bids.clear()
        self.asks.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_midpoint(self) -> Optional[float]:
        """Return ``(best_ask + best_bid) / 2``, or ``None`` if crossed or empty."""
        if not self.bids or not self.asks or self.is_crossed():
            return None
        return (max(self.bids) + min(self.asks)) / 2.0

    def get_spread(self) -> Optional[float]:
        """Return ``best_ask - best_bid``, or ``None`` if crossed or empty."""
        if not self.bids or not self.asks or self.is_crossed():
            return None
        return min(self.asks) - max(self.bids)

    def is_crossed(self) -> bool:
        """Return ``True`` if ``best_bid >= best_ask``."""
        if not self.bids or not self.asks:
            return False
        return max(self.bids) >= min(self.asks)

    @property
    def best_bid_qty(self) -> Optional[float]:
        """Quantity resting at the best bid, or ``None`` when empty."""
        if not self.bids:
            return None
        return self.bids[max(self.bids)]

    @property
    def best_ask_qty(self) -> Optional[float]:
        """Quantity resting at the best ask, or ``None`` when empty."""
        if not self.asks:
            return None
        return self.asks[min(self.asks)]


class ReplayEngine:
    """Stateful engine that replays NDJSON block envelopes market by market.

    One :class:`LimitOrderBook` and one :class:`OracleGate` are maintained per
    market.  The engine enforces the zero-corruption policy and returns
    block-close state dicts only when the oracle gate is open.

    Parameters
    ----------
    markets:
        List of market symbols to track.
    gate_threshold_bps:
        Forwarded to :class:`~oracle_gate.OracleGate`.
    gate_streak:
        Forwarded to :class:`~oracle_gate.OracleGate` as ``required_streak``.
    gate_gap_seconds:
        Forwarded to :class:`~oracle_gate.OracleGate` as ``gap_seconds``.
    """

    def __init__(
        self,
        markets: List[str],
        gate_threshold_bps: float = 25.0,
        gate_streak: int = 12,
        gate_gap_seconds: float = 30.0,
    ) -> None:
        self._books: Dict[str, LimitOrderBook] = {
            m: LimitOrderBook() for m in markets
        }
        self._gates: Dict[str, OracleGate] = {
            m: OracleGate(
                m,
                threshold_bps=gate_threshold_bps,
                required_streak=gate_streak,
                gap_seconds=gate_gap_seconds,
            )
            for m in markets
        }
        self._last_block: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_block(self, block_env: dict) -> Optional[dict]:
        """Process one block envelope and return a block-close state or ``None``.

        Returns ``None`` when the oracle gate is closed or when the block was
        discarded due to the zero-corruption policy.

        Parameters
        ----------
        block_env:
            Dict conforming to the block-envelope schema described in the
            module docstring.

        Returns
        -------
        Block-close state dict with keys ``timestamp``, ``market``,
        ``midpoint``, ``spread``, ``best_bid_qty``, ``best_ask_qty``,
        ``block_number``; or ``None``.
        """
        market: str = block_env["market"]
        block_number: int = block_env["block_number"]
        timestamp: float = block_env["timestamp"]
        updates: list = block_env.get("updates", [])
        oracle_price: Optional[float] = block_env.get("oracle_price")

        book = self._books.get(market)
        gate = self._gates.get(market)
        if book is None or gate is None:
            logger.debug("Unknown market %s – skipping block %d", market, block_number)
            return None

        # ── Zero-corruption policy: block-number rewind ──────────────────
        last_bn = self._last_block.get(market)
        if last_bn is not None and block_number <= last_bn:
            logger.warning(
                "Block rewind for %s: received %d, last was %d – resetting",
                market,
                block_number,
                last_bn,
            )
            self._reset_market(market)
            return None

        self._last_block[market] = block_number

        # ── Apply level diffs (negative-depth guard) ─────────────────────
        try:
            for upd in updates:
                book.update(upd["side"], float(upd["price"]), float(upd["qty"]))
        except ValueError as exc:
            logger.warning(
                "Negative depth on %s block %d: %s – resetting",
                market,
                block_number,
                exc,
            )
            self._reset_market(market)
            return None

        # ── Zero-corruption policy: crossed book ─────────────────────────
        if book.is_crossed():
            logger.warning(
                "Crossed book on %s block %d – resetting", market, block_number
            )
            self._reset_market(market)
            return None

        # ── Oracle gate update ────────────────────────────────────────────
        midpoint = book.get_midpoint()
        if midpoint is None:
            return None

        if oracle_price is not None:
            gate.update(timestamp, midpoint, float(oracle_price))

        if not gate.is_open():
            return None

        return {
            "timestamp": timestamp,
            "market": market,
            "midpoint": midpoint,
            "spread": book.get_spread(),
            "best_bid_qty": book.best_bid_qty,
            "best_ask_qty": book.best_ask_qty,
            "block_number": block_number,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_market(self, market: str) -> None:
        """Reset book and oracle gate for *market*; clear last block number."""
        self._books[market].reset()
        self._gates[market].reset()
        self._last_block.pop(market, None)
