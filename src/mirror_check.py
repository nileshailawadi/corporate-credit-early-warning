"""
Establish how the two circulating copies of this dataset differ.

Both copies carry the same 78,682 rows, 8,971 companies, 609 defaulting companies and the
same year distribution - every count matches exactly, so this is one dataset in two
packagings, not two datasets.

They differ in two ways, and both matter:

  1. COLUMN ORDER.  The Kaggle copy's X1..X18 are in the order its data dictionary
     describes.  The GitHub copy's are not.  Using the dictionary against the GitHub file
     silently mislabels every financial variable.

  2. MAGNITUDE.  The GitHub copy's accounting identities hold on ~56% of rows against
     ~100% on Kaggle, and every failure there resolves under a power-of-1000 rescaling of
     individual cells.  That is damage, not a different reporting convention.

The column permutation is recovered here without reference to either dictionary, using the
share of negative values per column as a fingerprint.  That statistic is invariant to any
rescaling, so it identifies the same variable across both copies even though the GitHub
magnitudes are broken.

This module is the reason `data.py` reads the Kaggle copy and will not quietly fall back.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from data import KAGGLE_MAP, load, _github


def rel(a, b, tol=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc = np.maximum(np.abs(a), np.abs(b)); sc = np.where(sc < 1e-9, 1.0, sc)
    return np.abs(a - b) / sc < tol


def _align(df, year_col):
    """Both copies carry identical (company, year) keys, so put them in the same row order."""
    d = df.rename(columns={year_col: 'fyear'}).copy()
    d['fyear'] = d.fyear.astype(int)
    return d.sort_values(['company_name', 'fyear']).reset_index(drop=True)


def _sign_zero_key(col):
    """Per-row sign and zero pattern, packed to bytes.

    Multiplying a cell by a positive power of 1,000 changes neither its sign nor whether
    it is zero, so this pattern is identical for the same variable in both copies even
    though the magnitudes in one of them are broken. It is a far sharper fingerprint than
    an aggregate share, which ties across every strictly-positive variable.
    """
    v = col.to_numpy(float)
    return (np.packbits(v < 0).tobytes(), np.packbits(v == 0).tobytes())


def _scale_match(a, b, tol=0.02):
    """Share of rows where a == b up to a power of 1,000 either way.

    The damage is a per-cell power-of-1000 fault, so the true counterpart column agrees
    row by row once that freedom is allowed, and an unrelated column does not.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0)
    hit = np.zeros(m.sum(), dtype=bool)
    for s in (1.0, 1e3, 1e6, 1e-3, 1e-6):
        hit |= rel(a[m], b[m] * s, tol)
    return float(hit.mean())


def recover_permutation(kaggle_raw, github_raw):
    """Match GitHub columns to Kaggle columns without using either data dictionary.

    Pass 1 - the exact per-row sign and zero pattern, which a power-of-1000 fault cannot
    change. This resolves every column that ever takes a negative or zero value.

    Pass 2 - for the strictly-positive columns, whose sign/zero pattern is uninformative,
    the share of rows matching up to a power of 1,000. Assigned greedily by best score
    among the columns pass 1 did not already claim.
    """
    X = [f'X{i}' for i in range(1, 19)]
    k = _align(kaggle_raw, 'year')
    g = _align(github_raw, 'fyear')
    assert (k.company_name.values == g.company_name.values).all(), 'row keys do not align'

    gk = {}
    for gc in X:
        gk.setdefault(_sign_zero_key(g[gc]), []).append(gc)

    out, claimed, pending = {}, set(), []
    for kc in X:
        cands = gk.get(_sign_zero_key(k[kc]), [])
        if len(cands) == 1:
            out[kc] = (cands[0], 'sign/zero pattern')
            claimed.add(cands[0])
        else:
            pending.append(kc)

    scored = sorted(((_scale_match(k[kc], g[gc]), kc, gc)
                     for kc in pending for gc in X if gc not in claimed), reverse=True)
    taken = set()
    for score, kc, gc in scored:
        if kc in taken or gc in claimed:
            continue
        out[kc] = (gc, f'value match {score:.3f}')
        claimed.add(gc); taken.add(kc)
    for kc in pending:
        out.setdefault(kc, ('?', 'unresolved'))
    return {kc: out[kc] for kc in X}


def identity_rates(d):
    return {
        'Revenue - OpEx = EBITDA': float(rel(d.total_revenue - d.total_opex, d.ebitda).mean()),
        'EBITDA - D&A = EBIT': float(rel(d.ebitda - d.dep_amort, d.ebit).mean()),
        'Revenue - COGS = GrossProfit': float(rel(d.total_revenue - d.cogs, d.gross_profit).mean()),
    }


def jump_rate(d):
    s = d.sort_values(['company_name', 'fyear'])
    r = s.groupby('company_name').total_assets.apply(lambda x: (x / x.shift()).dropna())
    r = r[np.isfinite(r) & (r > 0)]
    return float(((r > 100) | (r < 0.01)).mean())


if __name__ == '__main__':
    kag = load(with_division=False)
    kraw = pd.read_csv(__import__('data').KAGGLE_CSV)
    graw = _github()
    if graw is None:
        print('GitHub copy unavailable - clone https://github.com/sowide/bankruptcy_dataset '
              'into data/raw_sowide to run the comparison.')
        raise SystemExit

    print('SAME DATASET?')
    kk = set(zip(kraw.company_name, kraw.year.astype(int)))
    gg = set(zip(graw.company_name, graw.fyear.astype(int)))
    print(f'  rows                      {len(kraw):,} vs {len(graw):,}')
    print(f'  companies                 {kraw.company_name.nunique():,} vs '
          f'{graw.company_name.nunique():,}')
    print(f'  (company, year) keys identical   {kk == gg}')
    ks = kraw[['company_name', 'status_label']].drop_duplicates()
    gs = graw[['company_name', 'status_label']].drop_duplicates()
    m = ks.merge(gs, on='company_name', suffixes=('_k', '_g'))
    print(f'  status_label agrees on every firm {bool((m.status_label_k == m.status_label_g).all())}')

    print('\nCOLUMN PERMUTATION, recovered without using either data dictionary')
    perm = recover_permutation(kraw, graw)
    print(f'  {"kaggle":8s} {"github":8s} {"variable":22s} resolved by')
    for kc, (gc, how) in perm.items():
        print(f'  {kc:8s} {gc:8s} {KAGGLE_MAP[kc]:22s} {how}')
    identical = sum(kc == gc for kc, (gc, _) in perm.items())
    unresolved = sum(gc == '?' for gc, _ in perm.values())
    print(f'\n  columns sitting in the same position in both copies: {identical} of 18')
    print(f'  unresolved: {unresolved}')

    print('\nINTEGRITY, Kaggle copy')
    for k, v in identity_rates(kag).items():
        print(f'  {k:30s} {v:.4f}')
    print(f'  {"current assets <= total assets":30s} '
          f'{(kag.current_assets <= kag.total_assets).mean():.4f}')
    print(f'  {"1000x jumps in total assets":30s} {jump_rate(kag):.4f}')
    print(f'  {"median total assets":30s} {kag.total_assets.median():,.1f}')

    gmap = {kc: gc for kc, (gc, _) in perm.items()}
    gh = graw.rename(columns={gmap[kc]: KAGGLE_MAP[kc] for kc in KAGGLE_MAP})
    print('\nINTEGRITY, GitHub copy (under the recovered mapping, so the columns are named right)')
    for k, v in identity_rates(gh).items():
        print(f'  {k:30s} {v:.4f}')
    print(f'  {"current assets <= total assets":30s} '
          f'{(gh.current_assets <= gh.total_assets).mean():.4f}')
    print(f'  {"1000x jumps in total assets":30s} {jump_rate(gh):.4f}')
    print(f'  {"median total assets":30s} {gh.total_assets.median():,.1f}')

    print('\nIs the GitHub damage a power-of-1000 fault? Test: does every row that fails')
    print('EBITDA - D&A = EBIT satisfy it under SOME 1000^k rescaling of its cells?')
    import itertools
    S = [1.0, 1e3, 1e6, 1e-3]
    e, dp, eb = gh.ebitda.values, gh.dep_amort.values, gh.ebit.values
    ok = rel(e - dp, eb)
    base = ok.mean()
    for a, b, c in itertools.product(S, repeat=3):
        ok = ok | rel(e * a - dp * b, eb * c)
    print(f'  as distributed              {base:.4f}')
    print(f'  under some 1000^k rescaling {ok.mean():.4f}')
    print('\nConclusion: one dataset, two packagings. Use the Kaggle copy.')
