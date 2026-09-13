"""Credit feature library.

Ratios are what a credit analyst reads; raw dollar amounts are not comparable across a
panel spanning four orders of magnitude in firm size.  Trajectories matter as much as
levels - deterioration is the early-warning signal, the level is only context.
"""
from __future__ import annotations
import numpy as np, pandas as pd

EPS = 1e-6
TRAJECTORY = ['leverage', 're_ta', 'ebit_ta', 'current_ratio', 'ebitda_margin']


def build(d: pd.DataFrame, extended: bool = False):
    """
    Returns (features, panel) with the panel re-sorted to match the feature index.

    `extended=False` is the 25-feature core set every headline number in this repo was
    produced with.  `extended=True` adds efficiency ratios and industry-relative
    z-scores; it did NOT improve walk-forward performance (mean Gini 0.823 either way,
    worst fold 0.717 vs 0.786) and is kept for the W3 feature-selection work rather
    than because it helps.
    """
    d = d.sort_values(['company_name', 'fyear']).copy()
    ta = d.total_assets.abs() + EPS
    rev = d.total_revenue.abs() + EPS
    cl = d.current_liabilities.abs() + EPS
    tl = d.total_liabilities.abs() + EPS
    f = pd.DataFrame(index=d.index)

    # scale - sign-preserving log so the model can condition on firm size
    for c in ['total_assets', 'total_revenue', 'market_value', 'total_liabilities', 'ebitda']:
        f['log_' + c] = np.sign(d[c]) * np.log1p(d[c].abs())

    # leverage and coverage
    f['leverage'] = d.total_liabilities / ta
    f['lt_debt_ta'] = d.lt_debt / ta
    f['debt_ebitda'] = d.lt_debt / (d.ebitda.abs() + EPS)
    f['mve_tl'] = d.market_value / tl

    # liquidity
    f['current_ratio'] = d.current_assets / cl
    f['quick_ratio'] = (d.current_assets - d.inventory) / cl
    f['wc_ta'] = (d.current_assets - d.current_liabilities) / ta

    # profitability and accumulated equity
    f['re_ta'] = d.retained_earnings / ta
    f['ebit_ta'] = d.ebit / ta
    f['roa'] = d.net_income / ta
    f['ebitda_margin'] = d.ebitda / rev
    f['net_margin'] = d.net_income / rev
    f['gross_margin'] = d.gross_profit / rev

    f['asset_turnover'] = d.total_revenue / ta

    # Altman Z'' (emerging-market variant - no sales/TA term, so it travels across sectors)
    f['altman_z'] = 6.56 * f.wc_ta + 3.26 * f.re_ta + 6.72 * f.ebit_ta + 1.05 * f.mve_tl

    # trajectory - 1y and 2y change in the ratios that move before a default
    for c in TRAJECTORY:
        f[c + '_d1'] = f[c] - f[c].groupby(d.company_name).shift(1)
        f[c + '_d2'] = f[c] - f[c].groupby(d.company_name).shift(2)

    f['firm_age'] = d.fyear - d.groupby('company_name').fyear.transform('min')
    f['division'] = d.Division.astype('category').cat.codes

    if extended:
        f['days_receivable'] = 365 * d.receivables / rev
        f['days_inventory'] = 365 * d.inventory / (d.cogs.abs() + EPS)
        # industry-relative: 3x leverage is ordinary for a utility, lethal for software
        key = [d.Division.astype(str), d.fyear]
        for c in ('leverage', 'ebit_ta', 'current_ratio', 'ebitda_margin', 'altman_z'):
            g = f[c].groupby(key)
            iqr = g.transform(lambda s: s.quantile(.75) - s.quantile(.25))
            f[c + '_ind_z'] = (f[c] - g.transform('median')) / (iqr + EPS)

    return f.replace([np.inf, -np.inf], np.nan), d
