"""
W1 - Repair the power-of-1000 magnitude corruption in the American Bankruptcy panel.

Two independent sources of evidence pin the scale of each cell:

  (A) Within-firm continuity.  A company's total assets do not move by three orders
      of magnitude between consecutive fiscal years.  For each (company, variable)
      series we choose integer shifts k_t in {-2..2} minimising the total variation
      of log10|v_t| - 3*k_t, with a penalty on k != 0 so cells are left alone unless
      the series says otherwise.  Solved exactly by dynamic programming.

  (B) Accounting identities.  Revenue - OpEx = EBITDA, EBITDA - D&A = EBIT and
      Revenue - COGS = Gross profit tie seven P&L variables together inside a single
      row.  Residual rows are solved by exhaustive search over the small scale
      lattice, minimising the number of cells moved.

      Note on the third variable: `total_opex` is operating expense EXCLUDING
      depreciation and amortisation.  Revenue - OpEx reconciles to EBITDA (0.74 of
      rows before any identity solving), not to EBIT (0.04).  Treating this column as
      a full operating expense line - and the resulting margin as an EBIT margin - is
      a definitional error, independent of the scale corruption.

(A) fixes relative scale along time and covers all 18 variables; (B) fixes rows that
(A) cannot see - single-observation firms, and errors that shift a whole series.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from audit import load, MAP

VARS = list(MAP.values())
PNL = ['ebitda', 'dep_amort', 'ebit', 'total_revenue', 'cogs', 'gross_profit', 'total_opex']
KS = np.array([-2, -1, 0, 1, 2])          # exponent lattice, value is scaled by 1000^-k
LAMBDA = 0.35                              # penalty per unit |k|, in log10 units
TOL = 0.02                                 # relative tolerance for identity satisfaction


# --------------------------------------------------------------------------- (A)
def dp_shifts(z: np.ndarray) -> np.ndarray:
    """Integer shifts minimising TV(z - 3k) + LAMBDA*|k|. z is log10|v| (finite)."""
    T, K = len(z), len(KS)
    if T == 1:
        return np.zeros(1, dtype=int)
    pen = LAMBDA * np.abs(KS)
    cost = pen.copy()
    back = np.zeros((T, K), dtype=np.int8)
    for t in range(1, T):
        # step[a, b] = |(z_t - 3*KS[b]) - (z_{t-1} - 3*KS[a])|
        step = np.abs((z[t] - 3 * KS)[None, :] - (z[t - 1] - 3 * KS)[:, None])
        tot = cost[:, None] + step + pen[None, :]
        back[t] = np.argmin(tot, axis=0)
        cost = tot.min(axis=0)
    k = np.empty(T, dtype=int)
    j = int(np.argmin(cost))
    for t in range(T - 1, -1, -1):
        k[t] = KS[j]
        j = int(back[t, j])
    return k


def continuity_pass(d: pd.DataFrame) -> pd.DataFrame:
    """Per (company, variable) DP smoothing. Returns exponent frame aligned to d."""
    exp = pd.DataFrame(0, index=d.index, columns=VARS, dtype=np.int8)
    order = d.sort_values(['company_name', 'fyear']).index
    dd = d.loc[order]
    groups = dd.groupby('company_name', sort=False).indices
    pos = {ix: i for i, ix in enumerate(order)}
    for v in VARS:
        col = dd[v].to_numpy(float)
        out = np.zeros(len(col), dtype=np.int8)
        for _, idx in groups.items():
            vals = col[idx]
            m = np.isfinite(vals) & (np.abs(vals) > 0)
            if m.sum() < 2:
                continue
            k = dp_shifts(np.log10(np.abs(vals[m])))
            # anchor: the modal shift is defined as "no change", so we only ever
            # correct cells that disagree with the bulk of their own series
            k = k - np.bincount(k - KS.min(), minlength=len(KS)).argmax() - KS.min()
            sub = np.zeros(len(vals), dtype=np.int8)
            sub[m] = k
            out[idx] = sub
        exp[v] = pd.Series(out, index=order).reindex(d.index).to_numpy()
    return exp


# --------------------------------------------------------------------------- (B)
def _best(residual_stack: np.ndarray, combos: np.ndarray, prior: np.ndarray):
    """Pick, per row, the combo with smallest (violation, #cells moved, |k| total)."""
    viol = residual_stack > TOL                       # (C, N) bool
    score = viol * 1e6 + prior[:, None]               # prefer feasible, then fewest moves
    pick = np.argmin(score, axis=0)
    feasible = ~viol[pick, np.arange(viol.shape[1])]
    return combos[pick], feasible


def _rel(a, b):
    sc = np.maximum(np.abs(a), np.abs(b))
    sc = np.where(sc < 1e-9, 1.0, sc)
    return np.abs(a - b) / sc


def identity_pass(d: pd.DataFrame, exp: pd.DataFrame):
    """
    Solve remaining P&L scale errors row-wise, as a chain so that no step can break
    an identity an earlier step already satisfied:

        1. free (revenue, opex, ebitda)   ->  revenue - opex = ebitda
        2. free (dep_amort, ebit)         ->  ebitda - dep   = ebit      [ebitda pinned]
        3. free (cogs, gross_profit)      ->  revenue - cogs = gp        [revenue pinned]
    """
    import itertools
    v = {c: d[c].to_numpy(float) * (1000.0 ** -exp[c].to_numpy(float)) for c in PNL}
    exp = exp.copy()

    def solve(free, resid_fn):
        combos, stacks, priors = [], [], []
        for ks in itertools.product(KS, repeat=len(free)):
            trial = dict(v)
            for name, k in zip(free, ks):
                trial[name] = v[name] * (1000.0 ** -float(k))
            combos.append(ks)
            stacks.append(resid_fn(trial))
            priors.append(sum(k != 0 for k in ks))
        combos = np.array(combos); stacks = np.vstack(stacks); priors = np.array(priors, float)
        best, ok = _best(stacks, combos, priors)
        for j, name in enumerate(free):
            add = np.where(ok, best[:, j], 0).astype(np.int8)
            exp[name] = exp[name].to_numpy() + add
            v[name] = v[name] * (1000.0 ** -add.astype(float))
        return ok

    ok1 = solve(('total_revenue', 'total_opex', 'ebitda'),
                lambda t: _rel(t['total_revenue'] - t['total_opex'], t['ebitda']))
    ok2 = solve(('dep_amort', 'ebit'),
                lambda t: _rel(t['ebitda'] - t['dep_amort'], t['ebit']))
    ok3 = solve(('cogs', 'gross_profit'),
                lambda t: _rel(t['total_revenue'] - t['cogs'], t['gross_profit']))
    return exp, (ok1, ok2, ok3)


# --------------------------------------------------------------------------- report
def identity_rates(d):
    return {
        'Rev - OpEx = EBITDA':  float((_rel(d.total_revenue - d.total_opex, d.ebitda) < TOL).mean()),
        'EBITDA - D&A = EBIT':  float((_rel(d.ebitda - d.dep_amort, d.ebit) < TOL).mean()),
        'Rev - COGS = GrossPr': float((_rel(d.total_revenue - d.cogs, d.gross_profit) < TOL).mean()),
    }


def structural_rates(d):
    return {
        'current_assets<=total_assets':      float((d.current_assets <= d.total_assets).mean()),
        'current_liab<=total_liabilities':   float((d.current_liabilities <= d.total_liabilities).mean()),
        'inventory<=current_assets':         float((d.inventory <= d.current_assets).mean()),
        'receivables<=current_assets':       float((d.receivables <= d.current_assets).mean()),
        'lt_debt<=total_liabilities':        float((d.lt_debt <= d.total_liabilities).mean()),
    }


def jump_rate(d, col='total_assets'):
    s = d.sort_values(['company_name', 'fyear'])
    r = s.groupby('company_name')[col].apply(lambda x: (x / x.shift()).dropna())
    r = r[np.isfinite(r) & (r > 0)]
    return float(((r > 100) | (r < 0.01)).mean())


if __name__ == '__main__':
    d = load()
    before_id, before_st, before_jp = identity_rates(d), structural_rates(d), jump_rate(d)

    print('pass A - within-firm continuity (DP over integer shifts)')
    exp = continuity_pass(d)
    rep = d.copy()
    for c in VARS:
        rep[c] = d[c].to_numpy(float) * (1000.0 ** -exp[c].to_numpy(float))
    print(f'   cells moved: {(exp.to_numpy() != 0).mean():.4f}')
    mid_id = identity_rates(rep)

    print('pass B - accounting identities on residual rows')
    exp2, oks = identity_pass(d, exp)
    rep2 = d.copy()
    for c in VARS:
        rep2[c] = d[c].to_numpy(float) * (1000.0 ** -exp2[c].to_numpy(float))
    print(f'   cells moved (cumulative): {(exp2.to_numpy() != 0).mean():.4f}')
    print(f'   rows reconciled by identity solver: '
          f'{" ".join(f"{o.mean():.4f}" for o in oks)}')

    after_id, after_st, after_jp = identity_rates(rep2), structural_rates(rep2), jump_rate(rep2)

    print('\n' + '=' * 78)
    print(f'{"ACCOUNTING IDENTITY":36s} {"before":>9s} {"pass A":>9s} {"pass A+B":>9s}')
    for k in before_id:
        print(f'{k:36s} {before_id[k]:9.4f} {mid_id[k]:9.4f} {after_id[k]:9.4f}')
    print(f'\n{"STRUCTURAL INEQUALITY":36s} {"before":>9s} {"":>9s} {"after":>9s}')
    for k in before_st:
        print(f'{k:36s} {before_st[k]:9.4f} {"":>9s} {after_st[k]:9.4f}')
    print(f'\n{"1000x jumps in total_assets series":36s} {before_jp:9.4f} {"":>9s} {after_jp:9.4f}')
    print('=' * 78)

    rep2.to_parquet('outputs/clean_panel.parquet', index=False)
    exp2.to_parquet('outputs/scale_exponents.parquet', index=False)
    print('\nwrote outputs/clean_panel.parquet')
