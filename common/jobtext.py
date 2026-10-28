"""Portfolio job validation and the text format the CUDA engine reads."""

from __future__ import annotations

import math

MAX_UNDERLYINGS = 16
MAX_POSITIONS = 64
MIN_PATHS = 10_000
MAX_PATHS = 200_000_000


def sample_portfolio() -> dict:
    """A four-name demo book. Spots are illustrative, not live prices."""
    return {
        "seed": 42,
        "paths": 10_000_000,
        "bench_paths": 1_000_000,
        "horizon_days": 1,
        "underlyings": [
            {"name": "SPX", "spot": 5200.0, "vol": 0.16, "rate": 0.045, "dividend": 0.013, "drift": 0.032},
            {"name": "AAPL", "spot": 190.0, "vol": 0.28, "rate": 0.045, "dividend": 0.005, "drift": 0.040},
            {"name": "XOM", "spot": 110.0, "vol": 0.22, "rate": 0.045, "dividend": 0.032, "drift": 0.013},
            {"name": "GLD", "spot": 180.0, "vol": 0.14, "rate": 0.045, "dividend": 0.0, "drift": 0.045},
        ],
        "correlation": [
            [1.0, 0.55, 0.35, 0.05],
            [0.55, 1.0, 0.25, 0.02],
            [0.35, 0.25, 1.0, 0.10],
            [0.05, 0.02, 0.10, 1.0],
        ],
        "positions": [
            {"underlying": "SPX", "kind": "equity", "quantity": 8},
            {"underlying": "SPX", "kind": "put", "strike": 5000, "maturity_years": 0.25, "quantity": -4},
            {"underlying": "AAPL", "kind": "call", "strike": 200, "maturity_years": 0.50, "quantity": 30},
            {"underlying": "AAPL", "kind": "put", "strike": 170, "maturity_years": 0.50, "quantity": 20},
            {"underlying": "XOM", "kind": "equity", "quantity": 40},
            {"underlying": "GLD", "kind": "equity", "quantity": 25},
        ],
        "asian": {
            "underlying": "AAPL",
            "strike": 190.0,
            "maturity_years": 1.0,
            "steps": 32,
            "quantity": 1,
        },
    }


def _num(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return out


def _cholesky_ok(corr: list[list[float]]) -> bool:
    n = len(corr)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = corr[i][j]
            for k in range(j):
                total -= l[i][k] * l[j][k]
            if i == j:
                if total <= 1e-12:
                    return False
                l[i][j] = math.sqrt(total)
            else:
                l[i][j] = total / l[j][j]
    return True


def validate(job: dict) -> dict:
    """Return a normalized job or raise ValueError."""
    if not isinstance(job, dict):
        raise ValueError("job must be an object")

    underlyings = job.get("underlyings")
    if not isinstance(underlyings, list) or not underlyings:
        raise ValueError("underlyings must be a non-empty list")
    if len(underlyings) > MAX_UNDERLYINGS:
        raise ValueError(f"at most {MAX_UNDERLYINGS} underlyings")

    names = []
    norm_und = []
    for i, raw in enumerate(underlyings):
        if not isinstance(raw, dict):
            raise ValueError(f"underlying {i} must be an object")
        name = str(raw.get("name", "")).strip()
        if not name or len(name) > 31 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for c in name):
            raise ValueError(f"underlying {i} needs a short name of letters, numbers, '.', '_' or '-'")
        if name in names:
            raise ValueError(f"duplicate underlying {name}")
        spot = _num(raw.get("spot"), f"{name}.spot")
        vol = _num(raw.get("vol"), f"{name}.vol")
        rate = _num(raw.get("rate"), f"{name}.rate")
        dividend = _num(raw.get("dividend", 0.0), f"{name}.dividend")
        if "drift" in raw and raw["drift"] is not None:
            drift = _num(raw["drift"], f"{name}.drift")
        else:
            drift = rate - dividend
        if spot <= 0 or vol <= 0:
            raise ValueError(f"{name} spot and vol must be positive")
        if vol > 5 or abs(rate) > 1 or abs(dividend) > 1 or abs(drift) > 2:
            raise ValueError(f"{name} has a rate, dividend, drift, or vol outside the demo range")
        names.append(name)
        norm_und.append(
            {"name": name, "spot": spot, "vol": vol, "rate": rate, "dividend": dividend, "drift": drift}
        )

    n = len(norm_und)
    corr = job.get("correlation")
    if not isinstance(corr, list) or len(corr) != n:
        raise ValueError("correlation must be a square matrix matching the underlyings")
    norm_corr = []
    for i in range(n):
        row = corr[i]
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("correlation must be a square matrix matching the underlyings")
        out_row = []
        for j in range(n):
            c = _num(row[j], f"correlation[{i}][{j}]")
            if c < -1 or c > 1:
                raise ValueError("correlation entries must be between -1 and 1")
            out_row.append(c)
        norm_corr.append(out_row)
    for i in range(n):
        if abs(norm_corr[i][i] - 1.0) > 1e-8:
            raise ValueError("correlation diagonal must be 1")
        for j in range(i):
            if abs(norm_corr[i][j] - norm_corr[j][i]) > 1e-8:
                raise ValueError("correlation matrix must be symmetric")
    if not _cholesky_ok(norm_corr):
        raise ValueError("correlation matrix is not positive definite")

    paths = int(_num(job.get("paths", 10_000_000), "paths"))
    if paths < MIN_PATHS or paths > MAX_PATHS:
        raise ValueError(f"paths must be between {MIN_PATHS} and {MAX_PATHS}")
    bench = int(_num(job.get("bench_paths", min(paths, 1_000_000)), "bench_paths"))
    bench = max(MIN_PATHS, min(bench, paths, 5_000_000))
    seed = int(_num(job.get("seed", 42), "seed"))
    if seed < 0:
        raise ValueError("seed must be >= 0")
    horizon_days = _num(job.get("horizon_days", 1), "horizon_days")
    if horizon_days <= 0 or horizon_days > 252:
        raise ValueError("horizon_days must be in (0, 252]")
    horizon_years = horizon_days / 252.0

    positions = job.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("positions must be a non-empty list")
    if len(positions) > MAX_POSITIONS:
        raise ValueError(f"at most {MAX_POSITIONS} positions")
    norm_pos = []
    for i, raw in enumerate(positions):
        if not isinstance(raw, dict):
            raise ValueError(f"position {i} must be an object")
        name = str(raw.get("underlying", "")).strip()
        if name not in names:
            raise ValueError(f"position {i} references unknown underlying {name}")
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in ("equity", "call", "put"):
            raise ValueError(f"position {i} kind must be equity, call, or put")
        qty = _num(raw.get("quantity", 0), f"position {i} quantity")
        if qty == 0 or abs(qty) > 1e9:
            raise ValueError(f"position {i} quantity must be non-zero and reasonable")
        if kind == "equity":
            strike = 0.0
            maturity = 0.0
        else:
            strike = _num(raw.get("strike"), f"position {i} strike")
            maturity = _num(raw.get("maturity_years"), f"position {i} maturity")
            if strike <= 0 or maturity <= horizon_years:
                raise ValueError(
                    f"position {i} needs a positive strike and maturity longer than the VaR horizon"
                )
        norm_pos.append(
            {
                "underlying": name,
                "kind": kind,
                "strike": strike,
                "maturity_years": maturity,
                "quantity": qty,
            }
        )

    asian = None
    if job.get("asian"):
        raw = job["asian"]
        if not isinstance(raw, dict):
            raise ValueError("asian must be an object")
        name = str(raw.get("underlying", "")).strip()
        if name not in names:
            raise ValueError("asian references an unknown underlying")
        strike = _num(raw.get("strike"), "asian.strike")
        maturity = _num(raw.get("maturity_years"), "asian.maturity")
        steps = int(_num(raw.get("steps", 32), "asian.steps"))
        qty = _num(raw.get("quantity", 1), "asian.quantity")
        if strike <= 0 or maturity <= 0 or steps < 2 or steps > 256 or qty == 0:
            raise ValueError("asian needs strike > 0, maturity > 0, steps in [2, 256], quantity != 0")
        asian = {
            "underlying": name,
            "strike": strike,
            "maturity_years": maturity,
            "steps": steps,
            "quantity": qty,
        }

    return {
        "seed": seed,
        "paths": paths,
        "bench_paths": bench,
        "horizon_days": horizon_days,
        "underlyings": norm_und,
        "correlation": norm_corr,
        "positions": norm_pos,
        "asian": asian,
    }


def to_job_text(job: dict) -> str:
    job = validate(job)
    names = [u["name"] for u in job["underlyings"]]
    lines = [
        "MCJOB 1",
        f"seed {job['seed']}",
        f"paths {job['paths']}",
        f"bench_paths {job['bench_paths']}",
        f"horizon_days {job['horizon_days']:.16g}",
        f"underlyings {len(job['underlyings'])}",
    ]
    for u in job["underlyings"]:
        lines.append(
            f"{u['name']} {u['spot']:.16g} {u['vol']:.16g} {u['rate']:.16g} "
            f"{u['dividend']:.16g} {u['drift']:.16g}"
        )
    lines.append("correlation")
    for row in job["correlation"]:
        lines.append(" ".join(f"{v:.16g}" for v in row))
    lines.append(f"positions {len(job['positions'])}")
    for p in job["positions"]:
        lines.append(
            f"{names.index(p['underlying'])} {p['kind']} {p['strike']:.16g} "
            f"{p['maturity_years']:.16g} {p['quantity']:.16g}"
        )
    asian = job.get("asian")
    if asian:
        lines.append(
            f"asian {names.index(asian['underlying'])} {asian['strike']:.16g} "
            f"{asian['maturity_years']:.16g} {asian['steps']} {asian['quantity']:.16g}"
        )
    lines.append("")
    return "\n".join(lines)
