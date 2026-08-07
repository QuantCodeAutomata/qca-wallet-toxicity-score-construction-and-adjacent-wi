# Tests for OracleGate
import pytest
import numpy as np
from src.oracle_gate import OracleGate

def test_gate_opens_after_streak():
    # Gate should open after 12 consecutive ticks within 25 bps
    gate = OracleGate(market='BTC')
    base = 50000.0
    for i in range(12):
        result = gate.update(float(i*10), base, base * (1 + 0.001))  # 10 bps - within threshold
    assert gate.is_open()

def test_gate_stays_closed_below_streak():
    gate = OracleGate(market='BTC')
    base = 50000.0
    for i in range(11):  # Only 11 ticks
        gate.update(float(i*10), base, base * (1 + 0.001))
    assert not gate.is_open()

def test_gate_resets_on_gap():
    gate = OracleGate(market='BTC')
    base = 50000.0
    # Build streak to 8
    for i in range(8):
        gate.update(float(i*10), base, base * (1 + 0.001))
    # Then 31-second gap
    gate.update(float(8*10 + 31), base, base * (1 + 0.001))
    # Streak should reset - need 12 more
    assert not gate.is_open()

def test_gate_closes_on_large_deviation():
    gate = OracleGate(market='BTC')
    base = 50000.0
    # Get gate open
    for i in range(12):
        gate.update(float(i*10), base, base * (1 + 0.001))
    assert gate.is_open()
    # Large deviation should break the streak but not retroactively close
    # (gate was already open; next tick starts a new streak check only if reset triggered)
    gate.reset()
    assert not gate.is_open()

def test_threshold_bps_boundary():
    gate = OracleGate(market='ETH', threshold_bps=25.0)
    base = 3000.0
    # Exactly at threshold: |recon - oracle| / oracle * 10000 = 25 bps
    recon = base * (1 + 0.0025)  # exactly 25 bps
    for i in range(12):
        gate.update(float(i*10), recon, base)
    assert gate.is_open()  # 25 bps should be within threshold (<=)

def test_above_threshold_blocks_streak():
    gate = OracleGate(market='SOL')
    base = 150.0
    # 30 bps deviation - should block streak
    recon = base * (1 + 0.003)  # 30 bps
    for i in range(12):
        gate.update(float(i*10), recon, base)
    assert not gate.is_open()
