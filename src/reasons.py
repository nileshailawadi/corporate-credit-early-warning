"""
Reason codes, and a conceptual-soundness check.

A PD with no explanation cannot go to a credit committee.  This module turns SHAP
attributions into the sentences an analyst would write, and then asks the model a
question model risk management always asks (SR 11-7, conceptual soundness): does it
agree with credit intuition about the DIRECTION of each driver?

Where it disagrees, that is not automatically a bug - but it is automatically
something the model owner has to be able to explain, so it is printed rather than
buried.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
import lightgbm as lgb
from evaluate import folds, PARAMS, SEEDS

warnings.filterwarnings('ignore')

# label, value formatter, and the sign credit intuition expects on PD
#   +1  higher value -> higher default risk
#   -1  higher value -> lower default risk
#    0  no directional expectation
DRIVERS = {
    'leverage':        ('Total liabilities / assets',      '{:.2f}',  +1),
    'lt_debt_ta':      ('Long-term debt / assets',         '{:.2f}',  +1),
    'debt_ebitda':     ('Long-term debt / EBITDA',         '{:.1f}x', +1),
    'mve_tl':          ('Market value / total liabilities','{:.2f}x', -1),
    'current_ratio':   ('Current ratio',                   '{:.2f}',  -1),
    'quick_ratio':     ('Quick ratio',                     '{:.2f}',  -1),
    'wc_ta':           ('Working capital / assets',        '{:.2f}',  -1),
    're_ta':           ('Retained earnings / assets',      '{:.2f}',  -1),
    'ebit_ta':         ('EBIT / assets',                   '{:.2%}',  -1),
    'roa':             ('Return on assets',                '{:.2%}',  -1),
    'ebitda_margin':   ('EBITDA margin',                   '{:.1%}',  -1),
    'net_margin':      ('Net margin',                      '{:.1%}',  -1),
    'gross_margin':    ('Gross margin',                    '{:.1%}',  -1),
    'asset_turnover':  ('Asset turnover',                  '{:.2f}',  -1),
    'altman_z':        ('Altman Z-double-prime',           '{:.2f}',  -1),
    'days_receivable': ('Days receivable',                 '{:.0f}d', +1),
    'days_inventory':  ('Days inventory',                  '{:.0f}d', +1),
    'firm_age':        ('Years observed',                  '{:.0f}',   0),
    'division':        ('Industry division',               '{:.0f}',   0),
}
TREND = {'_d1': 'year on year', '_d2': 'over two years'}


def _ord(n: float) -> str:
    i = int(round(n))
    if 10 <= i % 100 <= 20:
        return f'{i}th'
    return f'{i}' + {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')


def describe(feature: str, value: float, pct: float | None = None) -> str:
    """Render one feature as a line an analyst would recognise."""
    suffix = next((s for s in TREND if feature.endswith(s)), None)
    base = feature[:-len(suffix)] if suffix else feature
    ind = base.endswith('_ind_z')
    if ind:
        base = base[:-len('_ind_z')]
    if base.startswith('log_'):
        label = base[4:].replace('_', ' ').capitalize() + ' (log scale)'
        fmt = '{:.1f}'
    else:
        label, fmt, _ = DRIVERS.get(base, (base.replace('_', ' ').capitalize(), '{:.2f}', 0))
    try:
        v = fmt.format(value)
    except (ValueError, TypeError):
        v = f'{value:.2f}'
    if suffix:
        move = 'down' if value < 0 else 'up'
        txt = f'{label} {move} {v.lstrip("-")} {TREND[suffix]}'
    elif ind:
        txt = f'{label} {v} ({value:+.1f} sd vs industry-year)'
    else:
        txt = f'{label} {v}'
    if pct is not None and not suffix:
        txt += f' — {_ord(pct)} pct of book'
    return txt


def reason_codes(shap_row, feat_row, pct_row, columns, k=3):
    """The k features pushing this obligor's PD up the most, most important first."""
    order = np.argsort(-shap_row)
    out = []
    for j in order[:k]:
        if shap_row[j] <= 0:
            break
        out.append(describe(columns[j], feat_row[j],
                            pct_row[j] if np.isfinite(pct_row[j]) else None))
    return out or ['No single driver dominates — risk is diffuse across the profile']


def direction_check(model, F, sample=4000, grid=9, seed=0):
    """One-way partial dependence: does PD move the way a credit analyst expects?"""
    rs = np.random.RandomState(seed)
    idx = rs.choice(len(F), min(sample, len(F)), replace=False)
    base = F.iloc[idx]
    rows = []
    for f, (label, _, expect) in DRIVERS.items():
        if f not in F.columns or expect == 0:
            continue
        qs = np.nanquantile(F[f], np.linspace(0.05, 0.95, grid))
        qs = np.unique(qs)
        if len(qs) < 3:
            continue
        pdp = []
        for q in qs:
            X = base.copy(); X[f] = q
            pdp.append(model.predict_proba(X)[:, 1].mean())
        pdp = np.array(pdp)
        # Spearman of PD against the feature grid. The magnitude matters as much as the
        # sign: |rho| below 0.5 means the curve is not monotone in either direction, and
        # calling that "disagrees" would be wrong - it is a different finding.
        rho = float(np.corrcoef(np.argsort(np.argsort(qs)),
                                np.argsort(np.argsort(pdp)))[0, 1])
        verdict = ('non-monotone' if abs(rho) < 0.5
                   else 'agrees' if np.sign(rho) == expect else 'CONTRADICTS')
        rows.append(dict(driver=label, feature=f, expected=int(expect),
                         rho=rho, verdict=verdict,
                         pd_low=pdp[0], pd_high=pdp[-1],
                         pd_range=pdp.max() - pdp.min()))
    return pd.DataFrame(rows)


def build_watchlist(d, F, start=2003, horizon=1, score_year=2018, seeds=SEEDS, top=40):
    """Fit on everything resolved before `score_year`, score that year, explain it."""
    import shap
    y = d[f'y_{horizon}y']
    tr = (d[f'elig_{horizon}y'] == 1) & (d.fyear >= start) & (d.fyear + horizon - 1 < score_year)
    te = (d[f'elig_{horizon}y'] == 1) & (d.fyear == score_year)

    models = [lgb.LGBMClassifier(**{**PARAMS, 'random_state': s}).fit(F[tr], y[tr])
              for s in seeds]
    p = np.mean([m.predict_proba(F[te])[:, 1] for m in models], axis=0)

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
        'fyear': score_year,
        'division': d.Division[te].to_numpy(),
        'pd': p,
        'actual_default': y[te].to_numpy(),
        'reason_1': [c[0] if len(c) > 0 else '' for c in codes],
        'reason_2': [c[1] if len(c) > 1 else '' for c in codes],
        'reason_3': [c[2] if len(c) > 2 else '' for c in codes]})
    out = out.sort_values('pd', ascending=False).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out, models[0], sv, Fte


if __name__ == '__main__':
    from features import build
    d = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)
    F, d = build(d, extended=True)
    F = F.reset_index(drop=True); d = d.reset_index(drop=True)

    wl, model, sv, Fte = build_watchlist(d, F)
    wl.to_csv('outputs/watchlist_2018.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 52)

    n_top = max(int(0.10 * len(wl)), 1)
    print(f'WATCHLIST — fiscal 2018, {len(wl):,} obligors, '
          f'{int(wl.actual_default.sum())} subsequently defaulted')
    print(f'top decile ({n_top} names) contains '
          f'{int(wl.head(n_top).actual_default.sum())} of them\n')
    print(wl.head(12)[['rank', 'company', 'division', 'pd', 'actual_default',
                       'reason_1', 'reason_2']].to_string(index=False))

    print('\n\nCONCEPTUAL SOUNDNESS — one-way partial dependence vs credit intuition')
    dc = direction_check(model, F.loc[Fte.index])
    dc = dc.sort_values('verdict')
    print(dc[['driver', 'expected', 'rho', 'verdict', 'pd_low', 'pd_high']]
          .round(4).to_string(index=False))
    n = dc.verdict.value_counts()
    print(f'\nagrees {n.get("agrees", 0)} · non-monotone {n.get("non-monotone", 0)} · '
          f'contradicts {n.get("CONTRADICTS", 0)}  (of {len(dc)})')
    bad = dc[dc.verdict == 'CONTRADICTS']
    if len(bad):
        print('\ncontradictions the model owner has to be able to explain:')
        for _, r in bad.iterrows():
            print(f'   {r.driver}: expected {"higher" if r.expected > 0 else "lower"} PD '
                  f'as it rises, observed the reverse '
                  f'(PD {r.pd_low:.4f} -> {r.pd_high:.4f}, rho {r.rho:+.2f})')
    dc.to_csv('outputs/direction_check.csv', index=False)
    print('\nwrote outputs/watchlist_2018.csv, outputs/direction_check.csv')
