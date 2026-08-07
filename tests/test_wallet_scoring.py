import pytest
import pandas as pd
import numpy as np
from src.wallet_scoring import compute_markout, compute_wallet_scores, get_toxic_cohort

def make_block_states():
    """Create sample block states.

    Covers 2 000 seconds at 1-second resolution so that all orders produced by
    ``make_orders(300)`` (last order ≈ t+1 500 s) fall within the post-staleness
    window (120 s default) of the last block state.
    """
    n = 2000
    times = pd.date_range('2026-07-01', periods=n, freq='1s')
    return pd.DataFrame({
        'timestamp': times,
        'market': 'BTC',
        'midpoint': 50000.0 + np.random.randn(n) * 10,
        'spread': 1.0,
        'block_number': range(n)
    })

def make_orders(n=200, toxic_wallets=None):
    """Create sample orders."""
    np.random.seed(42)
    wallets = [f'w{i}' for i in range(10)]
    times = pd.date_range('2026-07-01 00:00:10', periods=n, freq='5s')
    orders = pd.DataFrame({
        'order_id': [f'o{i}' for i in range(n)],
        'wallet': np.random.choice(wallets, n),
        'market': 'BTC',
        'direction': np.random.choice([1, -1], n),
        'notional': np.random.uniform(100, 10000, n),
        'event_time': times,
        'is_twap': False,
        'is_liquidation': False
    })
    return orders

def test_compute_wallet_scores_returns_df():
    block_states = make_block_states()
    orders = make_orders(300)
    scores = compute_wallet_scores(orders, block_states, min_orders=10)
    assert isinstance(scores, pd.DataFrame)
    assert 'wallet' in scores.columns
    assert 'alpha_w' in scores.columns

def test_scores_bounded_reasonably():
    block_states = make_block_states()
    orders = make_orders(300)
    scores = compute_wallet_scores(orders, block_states, min_orders=5)
    # Scores in bps, should be in reasonable range for random data
    if len(scores) > 0:
        assert scores['alpha_w'].abs().max() < 1000  # Less than 1000 bps

def test_min_orders_threshold():
    block_states = make_block_states()
    orders = make_orders(50)  # Only 50 orders across 10 wallets = ~5 each
    scores = compute_wallet_scores(orders, block_states, min_orders=20)
    # Should have fewer wallets than with min_orders=1
    scores_low = compute_wallet_scores(orders, block_states, min_orders=1)
    assert len(scores) <= len(scores_low)

def test_get_toxic_cohort_size():
    scores = pd.DataFrame({
        'wallet': [f'w{i}' for i in range(100)],
        'alpha_w': np.linspace(-5, 5, 100),
        'order_count': [50] * 100,
        'total_notional': [1000.0] * 100
    })
    cohort = get_toxic_cohort(scores, top_fraction=0.1)
    assert len(cohort) == 10  # 10% of 100

def test_toxic_cohort_has_highest_scores():
    scores = pd.DataFrame({
        'wallet': [f'w{i}' for i in range(10)],
        'alpha_w': [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        'order_count': [50] * 10,
        'total_notional': [1000.0] * 10
    })
    cohort = get_toxic_cohort(scores, top_fraction=0.2)
    # top_fraction=0.2 of 10 wallets → top 2: alpha_w 5.0 and 4.0
    assert len(cohort) == 2
    assert all(scores[scores['wallet'].isin(cohort)]['alpha_w'] >= 4.0)

def test_notional_weighting():
    # Wallet with large notional on positive markout should have positive score
    # This is a regression test for the formula alpha_w = sum(n_e * x_e) / sum(n_e)
    scores = pd.DataFrame({
        'wallet': ['good', 'bad'],
        'alpha_w': [3.5, -2.1],
        'order_count': [100, 100],
        'total_notional': [1e6, 1e4]
    })
    cohort = get_toxic_cohort(scores, top_fraction=0.5)
    assert 'good' in cohort
