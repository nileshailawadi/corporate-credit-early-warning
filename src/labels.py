"""
W2 - Point-in-time default labels, right-censoring, and walk-forward folds.

The dataset ships a company-constant `status_label`.  This module replaces it with
the label a credit model is actually fitted on: does THIS obligor default within the
next h years, judged only on what was observable at time t.

Label construction
------------------
The dataset authors label the fiscal year immediately preceding a Chapter 7/11
filing.  For a failed company that year is, by construction, its last observation,
so:

    default_year(i) = max fyear observed for company i, if i ever fails
    y_h(i, t)       = 1  iff  default_year(i) in [t, t + h - 1]

Censoring
---------
5,675 of 8,362 surviving companies leave the panel before 2018.  A company that
stops being observed is right-censored, not solvent forever - it was delisted,
acquired, went private, or simply dropped out of coverage.  For a non-defaulter the
outcome over [t, t+h-1] is only KNOWN if the company is observed through t+h-1:

    eligible_h(i, t) = 1  if i ever defaults                      (outcome always known)
                     = 1  if last_obs(i) >= t + h - 1             (observed through the window)
                     = 0  otherwise                               (dropped: outcome unknown)

Competing risks
---------------
The data cannot distinguish acquisition from delisting from loss of coverage, so
exits are pooled as one censoring event.  This is a real limitation: if exit is
correlated with credit quality in either direction, the estimated hazard is biased.
Treating exits as survivals - the alternative - is strictly worse, because it labels
an unknown outcome as a good one.
"""
from __future__ import annotations
import numpy as np, pandas as pd

PANEL_END = 2018
HORIZONS = (1, 2, 3)
RELIABLE_FROM = 2003   # see note at end of module


def build(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(['company_name', 'fyear']).copy()
    g = d.groupby('company_name')
    d['first_obs'] = g.fyear.transform('min')
    d['last_obs'] = g.fyear.transform('max')
    d['ever_fails'] = (d.status_label == 'failed').astype(int)

    # the fiscal year preceding the filing; NaN for survivors
    d['default_year'] = np.where(d.ever_fails == 1, d.last_obs, np.nan)

    # a surviving company that stops being observed before the panel ends is censored
    d['censored'] = ((d.ever_fails == 0) & (d.last_obs < PANEL_END)).astype(int)
    d['exit_year'] = np.where(d.ever_fails == 0, d.last_obs, np.nan)

    for h in HORIZONS:
        y = ((d.ever_fails == 1) &
             (d.default_year >= d.fyear) &
             (d.default_year <= d.fyear + h - 1)).astype(int)
        eligible = ((d.ever_fails == 1) | (d.last_obs >= d.fyear + h - 1)).astype(int)
        d[f'y_{h}y'] = y
        d[f'elig_{h}y'] = eligible

    # exit taxonomy - administrative censoring at the panel end is NOT an exit
    d['exit_type'] = np.where(
        d.fyear != d.last_obs, 'continuing',
        np.where(d.ever_fails == 1, 'default',
                 np.where(d.last_obs >= PANEL_END, 'administrative', 'censored')))

    # discrete-time hazard framing: time since first observation, event at default_year
    d['tenure'] = d.fyear - d.first_obs
    d['at_risk'] = 1
    d['event'] = d['y_1y']

    # label-capture reliability - see LABEL_CAPTURE note below
    d['label_reliable'] = (d.fyear >= RELIABLE_FROM).astype(int)
    return d


# Default capture in this panel is demonstrably incomplete before ~2003: over 1999-2002
# only 37 exits are labelled as defaults while 1,633 companies leave the panel, 52% of
# them with a negative Altman Z''.  The 37 labelled defaults of that era are also far
# less distressed (median Z'' -0.75, median ROA -0.001) than defaults from 2007 onward
# (median Z'' -6.8, median ROA -0.35), which is what an incomplete label set looks like:
# only the most unambiguous cases got captured.  Training across the boundary teaches
# the model that deeply distressed firms survive.


# --------------------------------------------------------------------- folds
def walk_forward(d, first_test=2012, last_test=PANEL_END, horizon=1, min_train=8,
                 obligor_disjoint=False):
    """
    Expanding-window folds.  Train on everything whose outcome window closes strictly
    before the test year, test on the single test year.  Yields (name, train_mask,
    test_mask).

    A row dated t with horizon h resolves at t + h - 1, so to avoid using an outcome
    that had not yet happened at the time of training we require t + h - 1 < T.
    """
    elig = d[f'elig_{horizon}y'] == 1
    for T in range(first_test, last_test + 1):
        tr = elig & (d.fyear + horizon - 1 < T) & (d.fyear >= d.fyear.min())
        te = elig & (d.fyear == T)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        if (d.fyear[tr].nunique() < min_train):
            continue
        if obligor_disjoint:
            seen = set(d.company_name[tr])
            te = te & ~d.company_name.isin(seen)
        if d[f'y_{horizon}y'][te].sum() == 0:
            continue
        yield f'{T}', tr, te


if __name__ == '__main__':
    from audit import load
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'outputs/clean_panel.parquet'
    d = build(pd.read_parquet(src))

    n_firms = d.company_name.nunique()
    print(f'panel {len(d):,} firm-years · {n_firms:,} companies')
    print(f'  ever fails          {d.groupby("company_name").ever_fails.first().sum():,}')
    print(f'  censored (exit<2018){d.groupby("company_name").censored.first().sum():,}')
    print(f'  observed to 2018    {(d.groupby("company_name").last_obs.first()==PANEL_END).sum():,}')

    print('\nLABEL YIELD BY HORIZON')
    print(f'{"h":>3s} {"eligible rows":>14s} {"defaults":>9s} {"rate":>8s} {"rows dropped":>13s}')
    for h in HORIZONS:
        e = d[f'elig_{h}y'] == 1
        print(f'{h:>3d} {e.sum():>14,} {d[f"y_{h}y"][e].sum():>9,} '
              f'{d[f"y_{h}y"][e].mean():>8.4f} {(~e).sum():>13,}')

    print('\nANNUAL DEFAULT HAZARD (1y label, eligible rows)')
    e = d.elig_1y == 1
    hz = d[e].groupby('fyear').y_1y.agg(['sum', 'size'])
    hz['rate'] = hz['sum'] / hz['size']
    hz.columns = ['defaults', 'at_risk', 'rate']
    print(hz.assign(rate=lambda x: (x.rate * 100).round(2)).to_string())

    print('\nEXIT REASON MIX BY YEAR (share of that year\'s exits that are defaults)')
    ex = d[d.fyear == d.last_obs].groupby('fyear').ever_fails.agg(['sum', 'size'])
    ex['default_share'] = (ex['sum'] / ex['size']).round(3)
    ex.columns = ['default_exits', 'all_exits', 'default_share']
    print(ex.to_string())

    d.to_parquet('outputs/labelled_panel.parquet', index=False)
    print('\nwrote outputs/labelled_panel.parquet')
