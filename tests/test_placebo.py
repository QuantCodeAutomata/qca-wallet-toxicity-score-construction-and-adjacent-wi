import pytest
import pandas as pd
import numpy as np
from src.placebo import bin_into_deciles, create_matching_cells, sample_placebo_cohort, compute_rank_pvalue

def test_decile_binning_produces_10_groups():
    values = pd.Series(np.random.randn(100))
    deciles = bin_into_deciles(values)
    assert len(deciles.unique()) <= 10

def test_decile_binning_length_preserved():
    values = pd.Series(range(100))
    deciles = bin_into_deciles(values)
    assert len(deciles) == 100

def test_matching_cells_adds_columns():
    df = pd.DataFrame({
        'wallet': [f'w{i}' for i in range(50)],
        'alpha_w': np.random.randn(50),
        'order_count': np.random.randint(100, 1000, 50),
        'total_notional': np.random.uniform(1e4, 1e6, 50)
    })
    df_cells = create_matching_cells(df)
    assert 'notional_decile' in df_cells.columns
    assert 'order_count_decile' in df_cells.columns

def test_placebo_cohort_size_matching():
    np.random.seed(0)
    # 100 non-toxic wallets
    non_toxic = pd.DataFrame({
        'wallet': [f'nt{i}' for i in range(100)],
        'alpha_w': -np.random.uniform(0, 5, 100),
        'order_count': np.random.randint(100, 1000, 100),
        'total_notional': np.random.uniform(1e4, 1e6, 100),
        'notional_decile': np.random.randint(0, 10, 100),
        'order_count_decile': np.random.randint(0, 10, 100)
    })
    toxic_cohort = [f'tx{i}' for i in range(10)]
    toxic_with_cells = pd.DataFrame({
        'wallet': toxic_cohort,
        'notional_decile': np.random.randint(0, 10, 10),
        'order_count_decile': np.random.randint(0, 10, 10)
    })
    cell_counts = toxic_with_cells.groupby(['notional_decile', 'order_count_decile']).size()
    rng = np.random.default_rng(42)
    placebo = sample_placebo_cohort(toxic_cohort, non_toxic, cell_counts, rng)
    assert len(placebo) <= len(toxic_cohort)
    assert all(w in non_toxic['wallet'].values for w in placebo)

def test_rank_pvalue_floor():
    real = 2.0
    placebo_increments = pd.Series(np.zeros(200))  # All 0 < 2.0
    p = compute_rank_pvalue(real, placebo_increments)
    assert abs(p - 1/201) < 1e-6

def test_rank_pvalue_ceiling():
    real = -100.0
    placebo_increments = pd.Series(np.ones(200))  # All 1.0 > -100
    p = compute_rank_pvalue(real, placebo_increments)
    assert p == 1.0  # (1+200)/201 = 1.0

def test_rank_pvalue_range():
    np.random.seed(0)
    real = 0.5
    placebo_increments = pd.Series(np.random.randn(200))
    p = compute_rank_pvalue(real, placebo_increments)
    assert 0 < p <= 1.0
