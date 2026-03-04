# generator.py
# ------------------------------
# Core generator for turning (bin_count) into concrete company sizes within each bin.
# This file does NOT do scaling to a global total yet.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from profiles import build_default_profiles, WeightFunc


# Standard KSH-like bins:
# 1–4, 5–9, 10–19, 20–49, 50–249, 250+
# Here we cap 250+ at 1000 (as you did previously).
BINS: Dict[str, Tuple[int, int]] = {
    "1-4": (1, 4),
    "5-9": (5, 9),
    "10-19": (10, 19),
    "20-49": (20, 49),
    "50-249": (50, 249),
    "250-1000": (250, 1000),
}


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return [w / total for w in weights]


def sample_int_in_bin(
    L: int,
    U: int,
    weight_func: WeightFunc,
    rng: random.Random,
) -> int:
    """
    Sample one integer in [L..U] using weights produced by weight_func(L, U).
    """
    weights = weight_func(L, U)
    probs = _normalize_weights(weights)

    # random.choices returns a list
    offset = rng.choices(range(0, U - L + 1), weights=probs, k=1)[0]
    return L + offset


def generate_company_sizes_for_bin(
    bin_name: str,
    count: int,
    profile: str = "decay_mild",
    rng: Optional[random.Random] = None,
    profiles: Optional[Dict[str, WeightFunc]] = None,
    probs_override: Optional[List[float]] = None,   # <-- ÚJ
) -> List[int]:
    if count < 0:
        raise ValueError("count must be >= 0")
    if count == 0:
        return []

    if bin_name not in BINS:
        raise KeyError(f"Unknown bin_name: {bin_name}. Known: {list(BINS.keys())}")

    if rng is None:
        rng = random.Random()

    L, U = BINS[bin_name]

    # Ha kaptunk előre kiszámolt eloszlást, azt használjuk (TEÁOR-specifikus simítás után)
    if probs_override is not None:
        if len(probs_override) != (U - L + 1):
            raise ValueError(f"probs_override length mismatch for {bin_name}")

        # 1) várható darabszámok
        expected = [count * float(p) for p in probs_override]

        # 2) alap kerekítés lefelé
        alloc = [int(x) for x in expected]
        missing = count - sum(alloc)

        # 3) a maradékot a legnagyobb törtrészek kapják (stabil, kevés zaj)
        frac = [(i, expected[i] - alloc[i]) for i in range(len(expected))]
        frac.sort(key=lambda t: t[1], reverse=True)
        for i in range(missing):
            alloc[frac[i][0]] += 1

        # 4) listává bontás
        out = []
        for i, a in enumerate(alloc):
            out.extend([L + i] * a)

        # 5) opcionálisan keverd meg, hogy ne legyen “tömbös”
        rng.shuffle(out)
        return out

    # különben a régi logika (profile + weight_func)
    if profiles is None:
        profiles = build_default_profiles()
    if profile not in profiles:
        raise KeyError(f"Unknown profile: {profile}. Known: {list(profiles.keys())}")

    weight_func = profiles[profile]
    return [sample_int_in_bin(L, U, weight_func, rng) for _ in range(count)]

BOUNDARIES = [
    ("1-4", "5-9"),
    ("5-9", "10-19"),
    ("10-19", "20-49"),
    ("20-49", "50-249"),
    ("50-249", "250-1000"),
]

def _base_probs_for_bin(bin_name: str, profile: str, profiles: Dict[str, WeightFunc]) -> List[float]:
    L, U = BINS[bin_name]
    w = profiles[profile](L, U)
    return _normalize_weights(w)

def _redistribute_mass(
    probs: List[float],
    idx_from: int,
    delta: float,
    idx_targets: List[int],
    base_probs: Optional[List[float]] = None,
) -> None:
    """
    probs[idx_from] -= delta, és a delta-t szétosztja a targets között.
    Ha base_probs meg van adva, akkor annak arányai szerint oszt (shape preservation).
    """
    if delta <= 0 or not idx_targets:
        return

    probs[idx_from] -= delta
    if probs[idx_from] < 0:
        delta += probs[idx_from]  # visszavonjuk a túlcsökkentést
        probs[idx_from] = 0.0
        if delta <= 0:
            return

    if base_probs is None:
        weights = [probs[i] for i in idx_targets]
    else:
        weights = [base_probs[i] for i in idx_targets]

    total = sum(weights)
    if total <= 0:
        share = delta / len(idx_targets)
        for i in idx_targets:
            probs[i] += share
        return

    for i, w in zip(idx_targets, weights):
        probs[i] += delta * (w / total)

def _smooth_one_boundary(
    prev_probs: List[float],
    next_probs: List[float],
    prev_total: int,
    next_total: int,
    threshold: float,
    prev_base: List[float],
    next_base: List[float],
    window: int = 2,
) -> None:
    """
    A prev utolsó pontja és a next első pontja között simít.
    threshold = 0.1 => max ~10x ugrás engedett, mert min/max >= 0.1.
    Csak a "nagyobb oldalt" vágjuk vissza, a többi értékre osztjuk szét.
    """
    if prev_total <= 0 or next_total <= 0:
        return  # üres bin -> nem piszkáljuk

    c_prev_edge = prev_total * prev_probs[-1]
    c_next_edge = next_total * next_probs[0]

    if c_prev_edge <= 0 or c_next_edge <= 0:
        return  # ha valamelyik 0, most nem erőltetjük (ahogy beszéltük)

    small = min(c_prev_edge, c_next_edge)
    large = max(c_prev_edge, c_next_edge)
    ratio = small / large  # 0..1

    if ratio >= threshold:
        return  # oké

    # cél: ratio == threshold  -> large_target = small / threshold
    large_target = small / threshold

    if c_prev_edge > c_next_edge:
        # prev edge túl nagy -> vágjuk vissza prev_probs[-1]-et
        delta_c = c_prev_edge - large_target
        delta_p = delta_c / prev_total
        delta_p = min(delta_p, prev_probs[-1] * 0.95)  # ne nullázzuk ki teljesen
        # ne vágjuk a prev edge-et a base-hez képest túl mélyre
        min_edge = 0.40 * prev_base[-1]  # CHANGEABLE
        max_delta_allowed = max(0.0, prev_probs[-1] - min_edge)
        delta_p = min(delta_p, max_delta_allowed)
        hi = len(prev_probs) - 1
        lo = max(0, hi - window)
        # LOW-PREFER targets: bin eleje (1,2,...)
        max_targets = min(window, len(prev_probs) - 1)
        targets = list(range(0, max_targets))
        _redistribute_mass(
            prev_probs,
            idx_from=hi,
            delta=delta_p,
            idx_targets=targets,
            base_probs=prev_base,
        )
    else:
        # next edge túl nagy -> vágjuk vissza next_probs[0]-át
        delta_c = c_next_edge - large_target
        delta_p = delta_c / next_total
        delta_p = min(delta_p, next_probs[0] * 0.95)
        lo = 1
        hi = min(len(next_probs), 1 + window)
        targets = list(range(lo, hi))  # első utáni window elem
        _redistribute_mass(
            next_probs,
            idx_from=0,
            delta=delta_p,
            idx_targets=targets,
            base_probs=next_base,
        )

@dataclass(frozen=True)
class CellKey:
    """
    Identifies one KSH cell (e.g. settlement, teaor, bin).
    You can extend this later.
    """
    settlement: str
    teaor: str
    bin_name: str


def generate_companies_from_counts(
    counts: Dict[CellKey, int],
    boundary_ratio_threshold: float = 0.35, # CHANGEABLE
    profile_by_teaor: Optional[Dict[str, str]] = None,
    profile_by_teaor_bin: Optional[Dict[tuple, str]] = None,
    default_profile: str = "decay_mild",
    seed: int = 12345,
) -> Dict[CellKey, List[int]]:
    """
    Main batch generator.

    counts:
        mapping from (settlement, teaor, bin) -> number of organizations (companies)

    profile_by_teaor:
        optional mapping teaor -> profile_name, allowing TEÁOR-dependent shapes.

    Returns:
        mapping from CellKey -> list of generated company sizes (length == counts[key])
    """
    rng = random.Random(seed)
    profiles = build_default_profiles()

    # ---- TEÁOR összesített bin-darabszámok (összes settlement együtt) ----
    teaor_bin_totals: Dict[str, Dict[str, int]] = {}
    for k, n in counts.items():
        teaor_bin_totals.setdefault(k.teaor, {}).setdefault(k.bin_name, 0)
        teaor_bin_totals[k.teaor][k.bin_name] += int(n)

    # ---- TEÁOR -> bin_name -> probs (simítva) ----
    teaor_bin_probs: Dict[str, Dict[str, List[float]]] = {}
    teaor_profile_used: Dict[str, str] = {}

    for teaor, bin_tot in teaor_bin_totals.items():
        chosen_profile = default_profile
        if profile_by_teaor is not None and teaor in profile_by_teaor:
            chosen_profile = profile_by_teaor[teaor]
            teaor_profile_used[teaor] = chosen_profile

        # base probs
        base_map: Dict[str, List[float]] = {bn: _base_probs_for_bin(bn, chosen_profile, profiles) for bn in BINS.keys()}
        probs_map: Dict[str, List[float]] = {bn: base_map[bn][:] for bn in BINS.keys()}  # másolat

        # határ-simítás: pár iteráció, hogy stabilabb legyen
        for _ in range(6): # CHANGEABLE
            for b_prev, b_next in BOUNDARIES:
                _smooth_one_boundary(
                    prev_probs=probs_map[b_prev],
                    next_probs=probs_map[b_next],
                    prev_total=bin_tot.get(b_prev, 0),
                    next_total=bin_tot.get(b_next, 0),
                    threshold=boundary_ratio_threshold,
                    prev_base=base_map[b_prev],
                    next_base=base_map[b_next],
                    window=2,  # CHANGEABLE
                )

        teaor_bin_probs[teaor] = probs_map

    out: Dict[CellKey, List[int]] = {}

    for key, n_orgs in counts.items():
        # 1) döntsd el, milyen profilt használ ez a cella
        chosen_profile = default_profile

        # (teaor, bin) override
        if profile_by_teaor_bin is not None:
            tb = (key.teaor, key.bin_name)
            if tb in profile_by_teaor_bin:
                chosen_profile = profile_by_teaor_bin[tb]

        # teaor override (csak ha nem volt bin-override)
        if chosen_profile == default_profile and profile_by_teaor is not None and key.teaor in profile_by_teaor:
            chosen_profile = profile_by_teaor[key.teaor]

        # 2) probs_override csak akkor menjen, ha ugyanazzal a profillal készült a teaor_bin_probs
        probs_override = None
        prof_used = teaor_profile_used.get(key.teaor, default_profile)
        if chosen_profile == prof_used:
            probs_override = teaor_bin_probs.get(key.teaor, {}).get(key.bin_name)

        # 3) egyszer generálj, és kész
        out[key] = generate_company_sizes_for_bin(
            bin_name=key.bin_name,
            count=n_orgs,
            profile=chosen_profile,
            rng=rng,
            profiles=profiles,
            probs_override=probs_override,
        )

    return out
