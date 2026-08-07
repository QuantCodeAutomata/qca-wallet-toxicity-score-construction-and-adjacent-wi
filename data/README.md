# Data

## What Data Is Needed

This project requires **Hyperliquid Level-4 raw streams** — the full, tick-by-tick feed of order-book events, trades (fills), and oracle price updates for all perpetual markets on the Hyperliquid exchange.

The specific data artefacts needed are:

| Stream type      | Content                                                                    | Granularity       |
|------------------|----------------------------------------------------------------------------|-------------------|
| `fills`          | Every taker/maker fill: wallet, side, notional, order ID, trade ID, flags  | Per-event         |
| `l2Book`         | Full Level-2 order-book snapshots (best bid/ask, depth)                    | Per-event (~ms)   |
| `oraclePrices`   | Hyperliquid oracle mid-price for each market                               | Per-block (~2 s)  |
| `blockSummaries` | Block metadata: timestamp, block number                                    | Per-block         |

---

## How to Obtain the Data

Hyperliquid does not offer a public historical data API for Level-4 streams. The data must be captured in real time from a **non-validating Hyperliquid node** that subscribes to the WebSocket feed.

Steps:

1. Run a non-validating node following the official Hyperliquid node documentation.
2. Subscribe to the relevant WebSocket channels (`fills`, `l2Book`, `oraclePrices`, `blockSummaries`).
3. Write incoming JSON messages to disk, partitioned by hour.
4. Compress each hourly file with `zstandard` (`.zst`).

Place the raw compressed files under `data/raw/` using the naming convention:

```
data/raw/<YYYY-MM-DD>/<stream_type>_<YYYY-MM-DD_HH>.ndjson.zst
```

---

## Expected Format

Each file is a **newline-delimited JSON** (NDJSON) stream compressed with `zstandard`. Each line is one JSON object representing a single event from the corresponding channel.

Example fill record:

```json
{
  "time": 1751234567890,
  "coin": "BTC",
  "side": "B",
  "px": "50123.5",
  "sz": "0.02",
  "hash": "0xabc...",
  "tid": 99001234,
  "users": ["0xWALLET_TAKER", "0xWALLET_MAKER"],
  "oid": [100001, 100002],
  "cloid": [null, null],
  "liquidation": false,
  "twap": false
}
```

---

## Synthetic Data for Testing

Because real Hyperliquid Level-4 data is not publicly available, **all unit tests use synthetically generated data**. The test fixtures in `tests/` create in-memory DataFrames that mimic the schema of parsed fills, block states, and order-book snapshots. No real data files are required to run the test suite.

To run the tests:

```bash
python -m pytest tests/ -v
```
