"""Data-integrity audit of the American Bankruptcy panel (Pellegrino et al., 1999-2018)."""
import pandas as pd, numpy as np

RAW = 'data/raw_sowide/american_bankruptcy_dataset.csv'
# Mapping recovered empirically by value-matching against the authors' own
# named supplementary files (financial_{train,validation,test}.csv).
# NOTE: this contradicts the labels published on the Kaggle mirror.
MAP = {'X1':'current_assets','X2':'total_assets','X3':'cogs','X4':'lt_debt',
       'X5':'dep_amort','X6':'ebit','X7':'ebitda','X8':'gross_profit','X9':'inventory',
       'X10':'current_liabilities','X11':'net_income','X12':'retained_earnings',
       'X13':'receivables','X14':'total_revenue','X15':'market_value',
       'X16':'total_liabilities','X17':'net_sales','X18':'total_opex'}

def load():
    d = pd.read_csv(RAW).rename(columns=MAP)
    d['fyear'] = d.fyear.astype(int)
    return d

def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc = np.maximum(np.abs(a), np.abs(b)); sc = np.where(sc < 1e-9, 1.0, sc)
    return np.abs(a - b) / sc

S = [1.0, 1e3, 1e6, 1e-3]

def scale_repair(df, cols, identity, tol=0.02):
    """Find per-cell powers of 1000 that satisfy `identity`. Returns exponent frame."""
    best = pd.DataFrame(np.nan, index=df.index, columns=cols)
    resid = np.full(len(df), np.inf)
    import itertools
    for combo in itertools.product(S, repeat=len(cols)):
        vals = {c: df[c].values * s for c, s in zip(cols, combo)}
        r = identity(vals)
        hit = (r < tol) & (r < resid)
        if hit.any():
            resid = np.where(hit, r, resid)
            for c, s in zip(cols, combo):
                best.loc[hit, c] = np.log10(s) / 3
    return best, resid

if __name__ == '__main__':
    d = load()
    print(f"panel: {len(d):,} firm-years | {d.company_name.nunique():,} companies | {d.fyear.min()}-{d.fyear.max()}")
    print(f"label: company-constant ({(d.groupby('company_name').status_label.nunique()>1).sum()} firms change label)")
    print()
    print("=== P&L identity: EBITDA - D&A = EBIT ===")
    cols = ['ebitda','dep_amort','ebit']
    exps, resid = scale_repair(d, cols, lambda v: rel(v['ebitda']-v['dep_amort'], v['ebit']))
    print(f"rows reconcilable under some 1000^k rescaling: {np.isfinite(resid).mean():.4f}")
    print(f"rows needing NO rescaling:                     {(exps==0).all(axis=1).mean():.4f}")
    print("\nper-cell exponent distribution (k in 1000^k):")
    print(exps.apply(lambda s: s.value_counts(normalize=True, dropna=False)).fillna(0).round(4).to_string())
    cell_corrupt = (exps.fillna(0) != 0).values.mean()
    print(f"\ncorrupted cells within this 3-variable block: {cell_corrupt:.4f}")
