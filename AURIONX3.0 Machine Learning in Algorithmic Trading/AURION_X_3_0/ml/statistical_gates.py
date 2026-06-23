"""AURION X 4.0 — Gates estadísticos avanzados (DSR + PBO).

Referencias:
    Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio.
        Journal of Portfolio Management, 40(5), 94–107.
        https://doi.org/10.3905/jpm.2014.40.5.094
    Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2016).
        The probability of backtest overfitting. Journal of Computational
        Finance, 20(4), 39–69. https://doi.org/10.21314/JCF.2016.322

Funciones puras, sin estado, sin I/O. Dependen de numpy + scipy.stats.
La integración con el pipeline existente se hace desde ``oos_validator.py``.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:  # scipy es opcional (grupo v4); fallback a CDF normal manual
    _HAS_SCIPY = False

from analytics.logger import get_logger

logger = get_logger(__name__)

EULER_MASCHERONI = 0.5772156649015329


def _normal_cdf(z: float) -> float:
    """CDF normal estándar. Usa scipy si está disponible, si no aproximación."""
    if _HAS_SCIPY:
        return float(_scipy_stats.norm.cdf(z))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _skewness(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurtosis_pearson(x: np.ndarray) -> float:
    """Kurtosis Pearson (no exceso). Normal = 3.0. Fisher = Pearson - 3."""
    n = x.size
    if n < 4:
        return 3.0
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma < 1e-12:
        return 3.0
    return float(((x - mu) ** 4).mean() / sigma ** 4)


def _sharpe_raw(returns: np.ndarray) -> float:
    """Sharpe sin anualizar (media / desv estándar de retornos por periodo)."""
    if returns.size < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    n_trials: int,
    sr_threshold_annual: float = 0.0,
    bars_per_year: int = 525600,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Corrige el Sharpe observado por:
      - Número de ensayos (selection bias / data dredging).
      - Skewness y kurtosis no-normales de la distribución de retornos.

    Args:
        returns: Retornos por barra (M1, hora, día — homogéneo).
        n_trials: Número de configuraciones probadas durante la búsqueda. A más
            ensayos, más penalización (mayor probabilidad de obtener un Sharpe
            alto por azar).
        sr_threshold_annual: Sharpe mínimo de referencia (anualizado). Normalmente 0.
        bars_per_year: Anualización. M1 24/7 ≈ 525_600. M5 trading 252d ≈ 19_656.
            Para XAUUSD M1 (24/7 minus weekends): se usa el aprox. 525_600.

    Returns:
        Diccionario con:
            ``sr_raw_per_bar``: Sharpe por barra.
            ``sr_annualized``: Sharpe anualizado.
            ``expected_max_sr``: SR máximo esperado bajo H0.
            ``dsr``: probabilidad de que el SR observado sea "verdadero" (≥0,
                ≤1). Buscamos DSR ≥ 0.60.
            ``skew``, ``kurt``, ``n_obs``, ``n_trials``.
    """
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 30:
        logger.warning("DSR: n_obs=%d < 30, devolviendo 0.0 (insuficiente).", n)
        return {
            "sr_raw_per_bar": 0.0,
            "sr_annualized": 0.0,
            "expected_max_sr": 0.0,
            "dsr": 0.0,
            "skew": 0.0,
            "kurt": 3.0,
            "n_obs": float(n),
            "n_trials": float(n_trials),
        }

    sr_bar = _sharpe_raw(arr)
    sr_annual = sr_bar * math.sqrt(bars_per_year)
    skew = _skewness(arr)
    kurt = _kurtosis_pearson(arr)

    # SR threshold expresado por barra
    sr_threshold_bar = sr_threshold_annual / math.sqrt(bars_per_year)

    # Expected max SR bajo H0 (Bailey 2014, ec. 5). El "Z_max" es el máximo
    # esperado de N normales estándar; debe escalarse por la desv. std del
    # estimador SR (≈ 1/sqrt(T-1)) para llevarlo a la misma escala que sr_bar.
    log_trials = math.log(max(n_trials, 1))
    if n_trials > 1 and log_trials > 0:
        z_max = math.sqrt(2.0 * log_trials) * (
            1.0 - EULER_MASCHERONI / math.sqrt(2.0 * log_trials)
        )
    else:
        z_max = 0.0
    sr_std = 1.0 / math.sqrt(max(n - 1, 1))
    expected_max_sr = sr_threshold_bar + z_max * sr_std

    # DSR (ecuación principal Bailey 2014)
    excess = (sr_bar - expected_max_sr) * math.sqrt(max(n - 1, 1))
    variance_term = 1.0 - skew * sr_bar + ((kurt - 1.0) / 4.0) * sr_bar ** 2
    if variance_term <= 0.0:
        variance_term = 1e-9
    z = excess / math.sqrt(variance_term)
    dsr = _normal_cdf(z)

    return {
        "sr_raw_per_bar": float(sr_bar),
        "sr_annualized": float(sr_annual),
        "expected_max_sr": float(expected_max_sr),
        "dsr": float(dsr),
        "skew": float(skew),
        "kurt": float(kurt),
        "n_obs": float(n),
        "n_trials": float(n_trials),
    }


def probability_backtest_overfitting(
    returns_matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    s: int = 16,
    max_combinations: int | None = 10_000,
) -> dict[str, float]:
    """Probability of Backtest Overfitting vía CSCV.

    Combinatorially Symmetric Cross-Validation (Bailey, Borwein, López de Prado
    y Zhu, 2016). Particiona la matriz de retornos (T x N estrategias) en ``s``
    submatrices y, para cada combinación de tamaño s/2, calcula el rank OOS de
    la mejor estrategia IS. PBO = probabilidad de que la mejor IS quede por
    debajo de la mediana OOS.

    Args:
        returns_matrix: array (T, N). Retornos por barra de cada una de las N
            estrategias candidatas. Para usar este test necesitamos N >= 2.
        s: Número de particiones (par). Por defecto 16.
        max_combinations: Tope de combinaciones para evitar combinatorial blow-up
            cuando s es grande (16 -> C(16,8) = 12870, ya tope ok). Para s=8 ->
            C(8,4)=70. Si la combinatoria supera ``max_combinations``, se
            muestrean aleatoriamente.

    Returns:
        Diccionario con ``pbo``, ``n_combinations``, ``n_strategies``, ``T``, ``s``.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"returns_matrix debe ser 2D, recibido shape={R.shape}")
    T, N = R.shape

    if N < 2:
        logger.warning("PBO: N=%d < 2, no se puede comparar (devuelvo 1.0).", N)
        return {"pbo": 1.0, "n_combinations": 0.0, "n_strategies": float(N), "T": float(T), "s": float(s)}

    if s % 2 != 0:
        raise ValueError(f"s debe ser par (recibido s={s}).")

    if T < s * 4:
        logger.warning("PBO: T=%d demasiado pequeño para s=%d (mín %d). Devolviendo 1.0.", T, s, s * 4)
        return {"pbo": 1.0, "n_combinations": 0.0, "n_strategies": float(N), "T": float(T), "s": float(s)}

    M = T // s
    fold_indices = [np.arange(i * M, (i + 1) * M) for i in range(s)]
    half = s // 2
    all_idx = list(range(s))

    # Jitter determinístico para romper ties (cuando varias estrategias tienen
    # SR esencialmente idéntico, np.argmax devolvería siempre el índice 0
    # sesgando el resultado del PBO). Magnitud despreciable vs el SR real.
    jitter = np.linspace(-1e-12, 1e-12, num=N)

    all_combos = list(combinations(all_idx, half))
    if max_combinations is not None and len(all_combos) > max_combinations:
        rng = np.random.default_rng(seed=42)
        sampled_pos = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in sampled_pos]

    overfit_count = 0
    total = 0
    for J in all_combos:
        J_list = list(J)
        Jc_list = [i for i in all_idx if i not in J_list]

        is_rows = np.concatenate([fold_indices[j] for j in J_list])
        os_rows = np.concatenate([fold_indices[j] for j in Jc_list])

        is_means = R[is_rows].mean(axis=0)
        is_stds = R[is_rows].std(axis=0, ddof=1)
        os_means = R[os_rows].mean(axis=0)
        os_stds = R[os_rows].std(axis=0, ddof=1)

        # Sharpe IS/OS (sin anualizar — basta el orden relativo). Aplicamos
        # un jitter determinístico para romper ties por aleatoriedad numérica.
        with np.errstate(divide="ignore", invalid="ignore"):
            sr_is = np.where(is_stds > 1e-12, is_means / is_stds, -np.inf)
            sr_os = np.where(os_stds > 1e-12, os_means / os_stds, -np.inf)
        # Reordena ties pseudo-aleatoriamente usando un offset determinístico
        # por combinación J (no afecta cuando las estrategias son realmente
        # distinguibles).
        rng_local = np.random.default_rng(seed=int(np.prod([j + 1 for j in J_list])) % (2**31 - 1))
        sr_is = sr_is + rng_local.uniform(-1e-12, 1e-12, size=N) + jitter
        sr_os = sr_os + rng_local.uniform(-1e-12, 1e-12, size=N) + jitter

        best_is = int(np.argmax(sr_is))
        # Rank de la mejor IS dentro del OS (1 = peor, N = mejor)
        os_order = np.argsort(sr_os)  # ascendente
        rank_in_os = int(np.where(os_order == best_is)[0][0]) + 1
        # Convertir a [0, 1]: rank/N. Si la mejor IS queda por debajo de la
        # mediana OS, hay overfit.
        relative_rank_os = rank_in_os / N

        if relative_rank_os <= 0.5:
            overfit_count += 1
        total += 1

    if total == 0:
        return {"pbo": 1.0, "n_combinations": 0.0, "n_strategies": float(N), "T": float(T), "s": float(s)}

    return {
        "pbo": float(overfit_count / total),
        "n_combinations": float(total),
        "n_strategies": float(N),
        "T": float(T),
        "s": float(s),
    }


def passes_statistical_gates(
    *,
    returns: Sequence[float],
    n_trials: int,
    returns_matrix: np.ndarray | None,
    min_dsr: float = 0.60,
    max_pbo: float = 0.30,
    bars_per_year: int = 525600,
) -> tuple[bool, str, dict[str, float]]:
    """Combina DSR y PBO en un único veredicto.

    Args:
        returns: retornos OOS del candidato.
        n_trials: número de configuraciones probadas (búsqueda hiperparams).
        returns_matrix: opcional (T, N) con histórico de N candidatos para PBO.
            Si es ``None`` (primer modelo, sin histórico), se omite el gate PBO.
        min_dsr: umbral mínimo de DSR.
        max_pbo: umbral máximo de PBO.

    Returns:
        ``(passes, reason, metrics_dict)``.
    """
    dsr_info = deflated_sharpe_ratio(
        returns, n_trials=n_trials, bars_per_year=bars_per_year
    )
    metrics: dict[str, float] = {f"dsr_{k}": v for k, v in dsr_info.items()}

    if dsr_info["dsr"] < min_dsr:
        return False, f"dsr_below_threshold:{dsr_info['dsr']:.3f}<{min_dsr:.2f}", metrics

    if returns_matrix is not None and np.asarray(returns_matrix).size > 0:
        pbo_info = probability_backtest_overfitting(returns_matrix)
        metrics.update({f"pbo_{k}": v for k, v in pbo_info.items()})
        if pbo_info["pbo"] > max_pbo:
            return False, f"pbo_above_threshold:{pbo_info['pbo']:.3f}>{max_pbo:.2f}", metrics
    else:
        metrics["pbo_skipped"] = 1.0

    return True, "ok", metrics
