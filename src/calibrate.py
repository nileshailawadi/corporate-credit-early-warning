"""
W5 - Calibration, and an agency-style master rating scale.

A score that ranks well is not a PD.  Ranking supports a watchlist; pricing a loan,
sizing a limit or feeding an expected-loss calculation needs the number to mean what it
says - a 3% PD has to default 3% of the time.  This module turns scores into
probabilities and probabilities into rating grades.

Calibration
-----------
Isotonic regression, fitted INSIDE each walk-forward fold on cross-validated training
predictions (CalibratedClassifierCV, cv=3).  Calibrating on the model's own in-sample
training scores would fit the overfitted part of the score distribution and look
excellent in-sample while failing out-of-time, which is the usual way this is done
wrong.

The master scale
----------------
21 notches, AAA to C, with geometric PD bands at a ratio of 1.5 per notch.  This is an
INTERNAL master scale wearing agency-style labels, not a claim that this model's BBB is
Standard & Poor's BBB - the labels are a communication device, and the bands come from
the scale design, not from any agency's published default study.  Saying otherwise
would be the kind of claim that collapses under one question in an interview.

Notches are attempted, as asked, and reported with per-grade default counts and Wilson
intervals so the reader can see exactly where the sample runs out.  Category-level
rollup (AAA/AA/A/BBB/BB/B/CCC-C) is where the statistics actually hold.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from evaluate import folds, PARAMS, SEEDS

warnings.filterwarnings('ignore')

# Geometric master scale, ratio 1.5 per notch. Upper bound of each grade's PD band.
NOTCHES = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-',
           'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C']
EDGES = np.array([1.0e-4, 1.5e-4, 2.3e-4, 3.5e-4, 5.3e-4, 8.0e-4, 1.2e-3, 1.8e-3,
                  2.7e-3, 4.1e-3, 6.2e-3, 9.3e-3, 1.4e-2, 2.1e-2, 3.2e-2, 4.8e-2,
                  7.2e-2, 1.08e-1, 1.62e-1, 2.43e-1])          # 20 edges -> 21 grades
CATEGORY = {n: ('CCC-C' if n.startswith(('CCC', 'CC', 'C')) and n != 'C' or n == 'C'
                else n.rstrip('+-')) for n in NOTCHES}


def grade(pd_hat: np.ndarray) -> np.ndarray:
    return np.array(NOTCHES)[np.digitize(pd_hat, EDGES)]


def wilson(k, n, z=1.96):
    """Wilson score interval - correct at the small counts a notch scale produces,
    unlike the normal approximation, which happily returns negative default rates."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


# --------------------------------------------------------------------- fit
def run(d, F, start=2003, horizon=1, seeds=SEEDS):
    """Walk-forward, returning one calibrated PD per test row per fold."""
    y = d[f'y_{horizon}y']
    out = []
    for T, tr, te in folds(d, start, horizon=horizon):
        ps = []
        for s in seeds:
            base = lgb.LGBMClassifier(**{**PARAMS, 'random_state': s})
            cal = CalibratedClassifierCV(base, method='isotonic', cv=3)
            cal.fit(F[tr], y[tr])
            ps.append(cal.predict_proba(F[te])[:, 1])
        out.append(pd.DataFrame({
            'fyear': T,
            'company': d.company_name[te].to_numpy(),
            'pd': np.mean(ps, axis=0),
            'y': y[te].to_numpy()}))
    r = pd.concat(out, ignore_index=True)
    r['grade'] = grade(r.pd.to_numpy())
    r['category'] = r.grade.map(CATEGORY)
    return r


# --------------------------------------------------------------------- diagnostics
def reliability(r, bins=10):
    q = pd.qcut(r.pd.rank(method='first'), bins, labels=False)
    g = r.groupby(q).agg(n=('y', 'size'), defaults=('y', 'sum'),
                         mean_pd=('pd', 'mean'), observed=('y', 'mean'))
    g['ratio'] = g.observed / g.mean_pd
    return g


def brier_decomposition(r, bins=10):
    """Brier = reliability - resolution + uncertainty (Murphy). Reliability is the
    calibration term: lower is better, zero is perfect."""
    q = pd.qcut(r.pd.rank(method='first'), bins, labels=False)
    obar = r.y.mean()
    rel = res = 0.0
    for _, g in r.groupby(q):
        nk, fk, ok = len(g), g.pd.mean(), g.y.mean()
        rel += nk * (fk - ok) ** 2
        res += nk * (ok - obar) ** 2
    n = len(r)
    return dict(brier=brier_score_loss(r.y, r.pd), reliability=rel / n,
                resolution=res / n, uncertainty=obar * (1 - obar))


def scale_table(r, col='grade', order=None):
    order = order or NOTCHES
    g = r.groupby(col).agg(obligor_years=('y', 'size'), defaults=('y', 'sum'),
                           mean_pd=('pd', 'mean'))
    g = g.reindex([o for o in order if o in g.index])
    g['observed'] = g.defaults / g.obligor_years
    ci = [wilson(k, n) for k, n in zip(g.defaults, g.obligor_years)]
    g['ci_lo'], g['ci_hi'] = [c[0] for c in ci], [c[1] for c in ci]
    g['share_of_book'] = g.obligor_years / g.obligor_years.sum()
    g['pd_in_ci'] = [(lo <= p <= hi) for p, lo, hi in zip(g.mean_pd, g.ci_lo, g.ci_hi)]
    return g


def migration(r):
    """Year-over-year grade transitions, category level."""
    a = r[['company', 'fyear', 'category']].copy()
    b = a.copy(); b['fyear'] = b.fyear - 1
    m = a.merge(b, on=['company', 'fyear'], suffixes=('_from', '_to'))
    cats = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC-C']
    t = pd.crosstab(m.category_from, m.category_to, normalize='index')
    return t.reindex(index=[c for c in cats if c in t.index],
                     columns=[c for c in cats if c in t.columns])


if __name__ == '__main__':
    from features import build
    d = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)
    F, d = build(d, extended=True)
    F = F.reset_index(drop=True); d = d.reset_index(drop=True)

    r = run(d, F)
    r.to_csv('outputs/calibrated_pd.csv', index=False)
    pd.set_option('display.width', 220)

    print(f'{len(r):,} obligor-years scored · {int(r.y.sum())} defaults · '
          f'observed rate {r.y.mean():.4%} · mean PD {r.pd.mean():.4%}')
    print(f'Gini {2 * roc_auc_score(r.y, r.pd) - 1:.4f}\n')

    print('BRIER DECOMPOSITION (Murphy)')
    for k, v in brier_decomposition(r).items():
        print(f'   {k:14s} {v:.6f}')

    print('\nRELIABILITY BY PREDICTED-PD DECILE')
    rel = reliability(r)
    print(rel.assign(mean_pd=lambda x: (x.mean_pd * 100).round(3),
                     observed=lambda x: (x.observed * 100).round(3),
                     ratio=lambda x: x.ratio.round(2)).to_string())

    print('\nMASTER SCALE - 21 NOTCHES')
    t = scale_table(r)
    print(t.assign(mean_pd=lambda x: (x.mean_pd * 100).round(3),
                   observed=lambda x: (x.observed * 100).round(3),
                   ci_lo=lambda x: (x.ci_lo * 100).round(3),
                   ci_hi=lambda x: (x.ci_hi * 100).round(3),
                   share_of_book=lambda x: (x.share_of_book * 100).round(1)).to_string())
    thin = (t.defaults < 5).sum()
    print(f'\nnotches with fewer than 5 defaults: {thin} of {len(t)} '
          f'({t.loc[t.defaults < 5, "obligor_years"].sum():,} obligor-years)')
    print(f'notches whose mean PD falls inside the observed Wilson interval: '
          f'{int(t.pd_in_ci.sum())} of {len(t)}')

    print('\nROLLED UP TO RATING CATEGORIES')
    tc = scale_table(r, 'category', ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC-C'])
    print(tc.assign(mean_pd=lambda x: (x.mean_pd * 100).round(3),
                    observed=lambda x: (x.observed * 100).round(3),
                    ci_lo=lambda x: (x.ci_lo * 100).round(3),
                    ci_hi=lambda x: (x.ci_hi * 100).round(3),
                    share_of_book=lambda x: (x.share_of_book * 100).round(1)).to_string())
    mono = tc.observed.is_monotonic_increasing
    print(f'\nobserved default rate monotonically increasing across categories: {mono}')

    print('\nONE-YEAR MIGRATION MATRIX (category, row-normalised)')
    print((migration(r) * 100).round(1).to_string())

    t.to_csv('outputs/master_scale_notches.csv')
    tc.to_csv('outputs/master_scale_categories.csv')
    print('\nwrote outputs/master_scale_{notches,categories}.csv, calibrated_pd.csv')
