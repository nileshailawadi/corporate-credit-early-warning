"""
Load the American Bankruptcy panel from the Kaggle distribution.

WHICH COPY YOU USE MATTERS. Two copies of this dataset are in circulation and they are
not interchangeable:

  Kaggle  kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset
          Columns in the order its data dictionary describes. Accounting identities hold
          on ~100% of rows. This is the copy to use.

  GitHub  github.com/sowide/bankruptcy_dataset
          Same 78,682 rows and 8,971 companies, same labels, same year distribution -
          but the 18 financial columns are in a DIFFERENT order, and individual cells
          have lost their magnitude by powers of 1,000. The identities hold on only 56%
          of rows there, and 100% of the failures resolve under 1000^k rescaling.

`mirror_check.py` establishes both facts. Anyone who clones the repo rather than
downloading from Kaggle silently gets the damaged copy with mislabelled columns, which
is how this was originally missed here.

The SIC Division is the one useful field the Kaggle copy lacks; it is joined from the
GitHub copy by company_name. That is safe - the identifiers, the (company, year) keys
and the status labels are identical across the two, and a categorical sector code cannot
be affected by a magnitude fault.
"""
from __future__ import annotations
import pathlib, subprocess
import pandas as pd

KAGGLE_MAP = {
    'X1': 'current_assets', 'X2': 'cogs', 'X3': 'dep_amort', 'X4': 'ebitda',
    'X5': 'inventory', 'X6': 'net_income', 'X7': 'receivables', 'X8': 'market_value',
    'X9': 'net_sales', 'X10': 'total_assets', 'X11': 'lt_debt', 'X12': 'ebit',
    'X13': 'gross_profit', 'X14': 'current_liabilities', 'X15': 'retained_earnings',
    'X16': 'total_revenue', 'X17': 'total_liabilities', 'X18': 'total_opex'}

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE_CSV = ROOT / 'data/kaggle/american_bankruptcy.csv'
GITHUB_DIR = ROOT / 'data/raw_sowide'
GITHUB_CSV = GITHUB_DIR / 'american_bankruptcy_dataset.csv'
GITHUB_URL = 'https://github.com/sowide/bankruptcy_dataset.git'

VARS = list(KAGGLE_MAP.values())


def _github(clone_if_missing=True) -> pd.DataFrame | None:
    if not GITHUB_CSV.exists() and clone_if_missing:
        try:
            subprocess.run(['git', 'clone', '--depth', '1', '-q', GITHUB_URL,
                            str(GITHUB_DIR)], check=True)
        except Exception:
            return None
    if not GITHUB_CSV.exists():
        return None
    return pd.read_csv(GITHUB_CSV)


def load(with_division=True) -> pd.DataFrame:
    if not KAGGLE_CSV.exists():
        raise FileNotFoundError(
            f'{KAGGLE_CSV} not found. Download american_bankruptcy.csv from\n'
            '  kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset\n'
            'and place it there. Do NOT substitute the GitHub copy - see the module '
            'docstring and mirror_check.py.')
    d = pd.read_csv(KAGGLE_CSV).rename(columns={**KAGGLE_MAP, 'year': 'fyear'})
    d['fyear'] = d.fyear.astype(int)

    if with_division:
        g = _github()
        if g is not None and 'Division' in g.columns:
            sic = g[['company_name', 'Division', 'MajorGroup']].drop_duplicates('company_name')
            d = d.merge(sic, on='company_name', how='left')
        else:
            d['Division'], d['MajorGroup'] = 'D', 0     # neutral fallback
    return d


if __name__ == '__main__':
    d = load()
    print(f'{len(d):,} firm-years · {d.company_name.nunique():,} companies · '
          f'{d.fyear.min()}-{d.fyear.max()}')
    print(f'defaulting companies: {d.loc[d.status_label == "failed", "company_name"].nunique()}')
    print(f'median total assets : {d.total_assets.median():,.1f}  (USD millions)')
    print(f'sector coverage     : {d.Division.notna().mean():.4f}')
    out = ROOT / 'outputs/clean_panel.parquet'
    out.parent.mkdir(exist_ok=True)
    d.to_parquet(out, index=False)
    print(f'wrote {out}')
