"""
W6 - Build the watchlist dataset the dashboard renders.

Scores the most recent fiscal year the panel carries (2018) with the full production
path: model fitted on the reliable window, isotonic calibration fitted inside the fold,
PDs mapped to the master scale, and SHAP reason codes attached to every name.

The PDs here are CALIBRATED, unlike the raw scores in reasons.py - a watchlist that
quotes a probability has to quote one that means something.
"""
from __future__ import annotations
import json, numpy as np, pandas as pd, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from evaluate import PARAMS, SEEDS
from tail_calibration import calibrated_fold, oot_multiplier, IsotonicOverlay
from calibrate import grade, CATEGORY
from reasons import reason_codes

warnings.filterwarnings('ignore')

SECTOR = {
    'A': 'Agriculture & forestry', 'B': 'Mining & extraction', 'C': 'Construction',
    'D': 'Manufacturing', 'E': 'Transport, utilities & telecom', 'F': 'Wholesale trade',
    'G': 'Retail trade', 'H': 'Finance, insurance & real estate', 'I': 'Services',
    'J': 'Public administration'}
SCORE_YEAR = 2018
START = 2003
TOP_N = 250


def build(d, F, score_year=SCORE_YEAR, start=START, seeds=SEEDS):
    import shap
    y = d.y_1y
    tr = (d.elig_1y == 1) & (d.fyear >= start) & (d.fyear < score_year)
    te = (d.elig_1y == 1) & (d.fyear == score_year)

    # Production calibration path: isotonic inside the fold, then the out-of-time
    # conservatism overlay estimated on the held-out prior year (see tail_calibration).
    ps, mults = [], []
    for s in seeds:
        base = calibrated_fold(F[tr], y[tr].to_numpy(), F[te], s)['isotonic']
        mult, _, raw = oot_multiplier(F, y, tr, score_year, s, fyear=d.fyear)
        mults.append((mult, raw))
        band = base >= IsotonicOverlay.BAND
        base[band] = np.clip(base[band] * mult, 0, 1 - 1e-7)
        ps.append(base)
    p = np.mean(ps, axis=0)
    applied = float(np.mean([m for m, _ in mults]))
    raw_est = float(np.nanmean([r for _, r in mults]))
    print(f'out-of-time overlay: applied {applied:.3f} '
          f'(raw estimate on FY{score_year - 1}: {raw_est:.3f}, floored at 1.0)')

    models = [lgb.LGBMClassifier(**{**PARAMS, 'random_state': s}).fit(F[tr], y[tr])
              for s in seeds]
    sv = np.mean([shap.TreeExplainer(m).shap_values(F[te]) for m in models], axis=0)
    if isinstance(sv, list):
        sv = sv[1]

    Fte = F[te]
    pct = Fte.rank(pct=True) * 100
    cols = list(F.columns)
    codes = [reason_codes(sv[i], Fte.iloc[i].to_numpy(), pct.iloc[i].to_numpy(), cols)
             for i in range(len(Fte))]

    out = pd.DataFrame({
        'company': d.company_name[te].to_numpy(),
        'sector': [SECTOR.get(x, x) for x in d.Division[te]],
        'pd': p,
        'grade': grade(p),
        'defaulted': y[te].to_numpy(),
        'r1': [c[0] if c else '' for c in codes],
        'r2': [c[1] if len(c) > 1 else '' for c in codes],
        'r3': [c[2] if len(c) > 2 else '' for c in codes]})
    out['category'] = out.grade.map(CATEGORY)
    out = out.sort_values('pd', ascending=False).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out


def dashboard_payload(wl):
    n = len(wl)
    k = max(int(0.10 * n), 1)
    cat_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC-C']
    dist = (wl.groupby('category').agg(n=('pd', 'size'), defaults=('defaulted', 'sum'),
                                       mean_pd=('pd', 'mean'))
            .reindex([c for c in cat_order if c in set(wl.category)]).fillna(0))
    sect = (wl.groupby('sector').agg(n=('pd', 'size'), defaults=('defaulted', 'sum'),
                                     mean_pd=('pd', 'mean'))
            .sort_values('mean_pd', ascending=False))
    top = wl.head(TOP_N)
    return {
        'scored_year': SCORE_YEAR,
        'n_obligors': int(n),
        'n_defaults': int(wl.defaulted.sum()),
        'decile_n': int(k),
        'decile_caught': int(wl.head(k).defaulted.sum()),
        'portfolio_pd': float(wl.pd.mean()),
        'observed_rate': float(wl.defaulted.mean()),
        'gini': float(2 * roc_auc_score(wl.defaulted, wl.pd) - 1),
        'categories': [{'category': i, 'n': int(r.n), 'defaults': int(r.defaults),
                        'mean_pd': float(r.mean_pd)} for i, r in dist.iterrows()],
        'sectors': [{'sector': i, 'n': int(r.n), 'defaults': int(r.defaults),
                     'mean_pd': float(r.mean_pd)} for i, r in sect.iterrows()],
        'names': [{'rank': int(r['rank']), 'company': r.company, 'sector': r.sector,
                   'pd': round(float(r.pd), 5), 'grade': r.grade,
                   'defaulted': int(r.defaulted),
                   'reasons': [x for x in (r.r1, r.r2, r.r3) if x]}
                  for _, r in top.iterrows()]}


if __name__ == '__main__':
    from features import build as build_features
    d = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)
    F, d = build_features(d, extended=True)
    F = F.reset_index(drop=True); d = d.reset_index(drop=True)

    wl = build(d, F)
    wl.to_csv('outputs/watchlist_calibrated.csv', index=False)
    payload = dashboard_payload(wl)
    with open('outputs/dashboard_data.json', 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

    print(f"FY{SCORE_YEAR} watchlist — {payload['n_obligors']:,} obligors, "
          f"{payload['n_defaults']} subsequent defaults")
    print(f"top decile ({payload['decile_n']}) catches {payload['decile_caught']} "
          f"({payload['decile_caught'] / payload['n_defaults']:.0%})")
    print(f"portfolio PD {payload['portfolio_pd']:.4%} vs observed "
          f"{payload['observed_rate']:.4%} · Gini {payload['gini']:.4f}\n")
    print(pd.DataFrame(payload['categories']).to_string(index=False))
    print()
    print(wl.head(10)[['rank', 'company', 'sector', 'grade', 'pd', 'defaulted', 'r1']]
          .to_string(index=False))
    import os
    print(f"\npayload {os.path.getsize('outputs/dashboard_data.json') / 1024:.0f} KB")
