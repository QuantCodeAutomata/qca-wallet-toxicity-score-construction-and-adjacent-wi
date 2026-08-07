import pytest
import pandas as pd
import numpy as np
from src.fills import pair_fills, group_aggressive_orders, filter_qualifying_orders

def make_sample_fills():
    """Create minimal sample fills dataframe for testing."""
    data = {
        'fill_time': pd.to_datetime(['2026-07-01 10:00:00'] * 4),
        'market': ['BTC', 'BTC', 'BTC', 'ETH'],
        'trade_id': ['t1', 't1', 't2', 't3'],
        'wallet': ['w1', 'maker_w1', 'w2', 'w3'],
        'side': ['buy', 'sell', 'sell', 'buy'],
        'role': ['taker', 'maker', 'taker', 'taker'],
        'notional': [1000.0, 1000.0, 500.0, 750.0],
        'order_id': ['o1', 'om1', 'o2', 'o3'],
        'is_twap': [False, False, False, False],
        'is_liquidation': [False, False, False, False]
    }
    data['day'] = pd.to_datetime(['2026-07-01'] * 4)
    return pd.DataFrame(data)

def test_pair_fills_basic():
    fills = make_sample_fills()
    paired = pair_fills(fills)
    assert len(paired) >= 1
    assert 'taker_wallet' in paired.columns or 'wallet' in paired.columns

def test_group_aggressive_orders():
    fills = make_sample_fills()
    taker_fills = fills[fills['role'] == 'taker'].copy()
    orders = group_aggressive_orders(taker_fills)
    assert 'wallet' in orders.columns
    assert 'direction' in orders.columns
    assert 'notional' in orders.columns
    assert set(orders['direction'].unique()).issubset({1, -1})

def test_filter_qualifying_orders():
    df = pd.DataFrame({
        'wallet': ['w1', 'w2', 'w3'],
        'direction': [1, -1, 1],
        'notional': [100.0, 200.0, 300.0],
        'is_twap': [False, True, False],
        'is_liquidation': [False, False, True]
    })
    filtered = filter_qualifying_orders(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]['wallet'] == 'w1'

def test_direction_encoding():
    fills = make_sample_fills()
    taker_fills = fills[fills['role'] == 'taker'].copy()
    orders = group_aggressive_orders(taker_fills)
    assert all(d in {1, -1} for d in orders['direction'])

def test_empty_fills():
    empty = pd.DataFrame(columns=['fill_time', 'market', 'trade_id', 'wallet', 'side', 'role', 'notional', 'order_id', 'is_twap', 'is_liquidation', 'day'])
    orders = group_aggressive_orders(empty)
    assert len(orders) == 0
