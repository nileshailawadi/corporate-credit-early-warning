"""Verification: does the W1 repair change anything that matters?

Three checks:
  1. Altman Z'' as a standalone discriminator, raw vs repaired (no fitting involved -
     if the repair is real, the 1968 formula should separate defaulters better on it).
  2. The full Arm D model (12-month PD, obligors held out, out-of-time), raw vs repaired.
  3. Cross-sectional plausibility of the headline ratios against known corporate norms.
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from audit import load
from leakage import features
warnings.filterwarnings('ignore')
RNG = 42


def labels(d):
    last = d.groupby('company_name').fyear.transform('max')
    y = ((d.status_label == 'failed') & (d.fyear == last)).astype(int)
    censored = (d.status_label == 'alive') & (d.fyear == last) & (last < 2018)
    return y, ~censored


def arm_d(d, tag):
    F, d2 = features(d)
    y, keep = labels(d2)
    tr, te = (d2.fyear <= 2011) & keep, (d2.fyear >= 2015) & keep
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                           random_state=RNG, verbose=-1).fit(F[tr], y[tr])
    p = m.predict_proba(F[te])[:, 1]
    yt = y[te].to_numpy()
    k = max(int(0.10 * len(p)), 1)
    auc = roc_auc_score(yt, p)
    return dict(panel=tag, gini=2 * auc - 1, pr_auc=average_precision_score(yt, p),
                top_decile=float(yt[np.argsort(-p)[:k]].sum() / yt.sum()))


def altman(d, tag):
    """Z'' = 6.56*WC/TA + 3.26*RE/TA + 6.72*EBIT/TA + 1.05*MVE/TL. Lower = riskier."""
    eps = 1e-6
    ta = d.total_assets.abs() + eps
    z = (6.56 * (d.current_assets - d.current_liabilities) / ta
         + 3.26 * d.retained_earnings / ta
         + 6.72 * d.ebit / ta
         + 1.05 * d.market_value / (d.total_liabilities.abs() + eps))
    y, keep = labels(d)
    m = keep & np.isfinite(z) & (d.fyear >= 2015)
    auc = roc_auc_score(y[m], -z[m])
    return dict(panel=tag, altman_gini=2 * auc - 1,
                median_z=float(z[m].median()),
                pct_z_below_minus5=float((z[m] < -5).mean()))


if __name__ == '__main__':
    raw = load()
    rep = pd.read_parquet('outputs/clean_panel.parquet')

    print('1. ALTMAN Z-DOUBLE-PRIME, unfitted, 2015-2018')
    a = pd.DataFrame([altman(raw, 'raw panel'), altman(rep, 'repaired panel')]).set_index('panel')
    print(a.round(4).to_string(), '\n')

    print('2. ARM D - 12-month PD, obligors held out, out-of-time')
    b = pd.DataFrame([arm_d(raw, 'raw panel'), arm_d(rep, 'repaired panel')]).set_index('panel')
    print(b.round(4).to_string(), '\n')

    print('3. CROSS-SECTIONAL PLAUSIBILITY (median, all firm-years)')
    eps = 1e-6
    rows = []
    for tag, d in [('raw panel', raw), ('repaired panel', rep)]:
        ta = d.total_assets.abs() + eps
        rows.append({
            'panel': tag,
            'total_liab/TA': float((d.total_liabilities / ta).median()),
            'current_ratio': float((d.current_assets / (d.current_liabilities.abs() + eps)).median()),
            'revenue/TA': float((d.total_revenue / ta).median()),
            'gross_margin': float((d.gross_profit / (d.total_revenue.abs() + eps)).median()),
        })
    print(pd.DataFrame(rows).set_index('panel').round(3).to_string())
    print('\nreference: US listed non-financials typically sit near TL/TA 0.5,'
          '\ncurrent ratio 1.5-2.0, revenue/TA 0.7-1.0, gross margin 0.30-0.40.')

    pd.concat([a, b], axis=1).to_csv('outputs/repair_verification.csv')
