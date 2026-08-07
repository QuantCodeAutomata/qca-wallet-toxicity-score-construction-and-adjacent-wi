import pytest
import pandas as pd
import numpy as np
from src.features import build_100ms_grid, build_targets

def make_states(coin='BTC', n=500):
    times = pd.date_range('2026-07-11', periods=n, freq='200ms')
    return pd.DataFrame({
        'timestamp': times,
        'market': coin,
        'midpoint': 50000.0 + np.cumsum(np.random.randn(n) * 0.5),
        'spread': 1.0,
        'best_bid_qty': np.random.uniform(0.5, 5, n),
        'best_ask_qty': np.random.uniform(0.5, 5, n),
    })

def test_grid_100ms_frequency():
    start = pd.Timestamp('2026-07-11 10:00:00')
    end = pd.Timestamp('2026-07-11 10:01:00')  # 1 minute = 600 points
    grid = build_100ms_grid(start, end)
    diffs = pd.Series(grid).diff().dropna()
    assert all(diffs == pd.Timedelta('100ms'))

def test_targets_shape():
    states = make_states()
    grid = pd.DatetimeIndex(states['timestamp'])
    targets = build_targets(grid, states, 'BTC', horizons_s=[1.0, 5.0, 10.0])
    assert '1.0s' in targets.columns or 1.0 in targets.columns or 'h_1.0' in targets.columns

def test_targets_missing_day_boundary():
    # Targets crossing day boundary should be NaN
    states = make_states(n=2000)
    # Put timestamps near midnight
    times = pd.date_range('2026-07-11 23:59:55', periods=200, freq='100ms')
    states_boundary = pd.DataFrame({
        'timestamp': times,
        'market': 'BTC',
        'midpoint': 50000.0 + np.cumsum(np.random.randn(200) * 0.1),
        'spread': 1.0,
        'best_bid_qty': 1.0,
        'best_ask_qty': 1.0,
    })
    grid = pd.DatetimeIndex(states_boundary['timestamp'])
    targets = build_targets(grid, states_boundary, 'BTC', horizons_s=[10.0])
    # Some should be NaN (those that would need next day's midpoint)
    assert targets.isna().any().any()

def test_grid_produces_timestamps():
    start = pd.Timestamp('2026-07-11')
    end = pd.Timestamp('2026-07-11 00:01:00')
    grid = build_100ms_grid(start, end)
    assert len(grid) > 0
    assert isinstance(grid, pd.DatetimeIndex)
