# profiles.py
# ------------------------------
# Weight-profiles for sampling integer company sizes within a bin [L..U].
# The output weights do NOT have to sum to 1; generator normalizes them.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


WeightFunc = Callable[[int, int], List[float]]


def _validate_bounds(L: int, U: int) -> None:
    if not (isinstance(L, int) and isinstance(U, int)):
        raise TypeError("L and U must be integers")
    if L > U:
        raise ValueError(f"Invalid bounds: L={L} > U={U}")


def uniform_weights(L: int, U: int) -> List[float]:
    """
    Uniform weights within [L..U].
    """
    _validate_bounds(L, U)
    n = U - L + 1
    return [1.0] * n


def power_decay_weights(L: int, U: int, alpha: float = 1.0, floor: float = 0.25) -> List[float]:
    """
    Monotonically decreasing weights with a minimum floor to avoid spikes at L.
    floor in [0..1] mixes some uniform mass into the distribution.
    """
    _validate_bounds(L, U)
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if not (0.0 <= floor <= 1.0):
        raise ValueError("floor must be between 0 and 1")

    n = U - L + 1

    base = [1.0 / ((k) ** alpha) for k in range(1, n + 1)]
    # mix with uniform to avoid extreme minima
    mixed = [(1.0 - floor) * w + floor * 1.0 for w in base]
    return mixed


def exp_decay_weights(L: int, U: int, lam: float = 0.20) -> List[float]:
    """
    Exponential-like decay (discrete), favoring smaller sizes.

    weight(i) = exp(-lam * i), where i = 0..(U-L)
    Larger lam => steeper decay.
    """
    _validate_bounds(L, U)
    if lam < 0:
        raise ValueError("lam must be >= 0")

    import math
    n = U - L + 1
    return [math.exp(-lam * i) for i in range(n)]


@dataclass(frozen=True)
class Profile:
    """
    A named profile holding a weight function and its parameters.
    """
    name: str
    func: WeightFunc


def build_default_profiles() -> Dict[str, WeightFunc]:
    """
    Returns a registry of profiles you can reference by name.
    """
    return {
        "uniform": uniform_weights,
        # Power-decay variants (simple + very robust):
        "decay_mild": lambda L, U: power_decay_weights(L, U, alpha=0.7, floor=0.35),
        "decay_strong": lambda L, U: power_decay_weights(L, U, alpha=1.2, floor=0.25),
        #"decay_mild": lambda L, U: power_decay_weights(L, U, alpha=0.9, floor=0.30),
        #"decay_strong": lambda L, U: power_decay_weights(L, U, alpha=1.4, floor=0.22),
        # Exponential variants:
        "exp_mild": lambda L, U: exp_decay_weights(L, U, lam=0.15),
        "exp_strong": lambda L, U: exp_decay_weights(L, U, lam=0.35),
        # Extra-strong variants for edge smoothing:
        "decay_ultra": lambda L, U: power_decay_weights(L, U, alpha=2.0, floor=0.18),
        "exp_ultra": lambda L, U: exp_decay_weights(L, U, lam=0.60),
    }
