"""
External validation of Finding 04, against an authoritative filings register.

Finding 04 - that default capture in this panel is incomplete before 2003 - rested
entirely on internal evidence: the shape of the hazard, and the fact that early exits
look distressed while carrying survival labels.  That is an inference.  This module
turns it into a measurement by comparing the panel's defaults against US Courts
business Chapter 11 filing statistics, which are a complete administrative count.

Universe mismatch, stated plainly: US Courts counts every business Chapter 11 filer,
private companies included - tens of thousands a year against a few dozen in this panel
of listed firms.  The absolute ratio is therefore meaningless.  What is NOT meaningless
is its time profile.  If the panel captures defaults at a stable rate, panel defaults
should track the national series up to a constant, and the implied capture rate should
be flat.  If capture is incomplete in some years, the implied rate dips in exactly
those years.

Source: rdinter/historical-bankruptcies, tidied from the US Courts F-2 tables.
Series: BCHAP_11, quarterly, aggregated to calendar years, 2001-2018.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

COURTS = 'data/hb/1-tidy/bankruptcy/national_quarterly_all.csv'
PANEL = 'outputs/labelled_panel.parquet'
CALIB_FROM, CALIB_TO = 2004, 2018     # years assumed to have reliable capture
TEST_YEARS = (2001, 2002, 2003)       # years under suspicion


def national() -> pd.Series:
    d = pd.read_csv(COURTS, parse_dates=['DATE'])
    d['filing_year'] = d.DATE.dt.year
    s = d.groupby('filing_year').BCHAP_11.sum()
    full = d.groupby('filing_year').size() == 4      # drop partial years
    return s[full].rename('national_ch11')


def panel() -> pd.DataFrame:
    d = pd.read_parquet(PANEL)
    e = d[d.elig_1y == 1]
    g = e.groupby('fyear').agg(panel_defaults=('y_1y', 'sum'), at_risk=('y_1y', 'size'))
    # the dataset labels the fiscal year BEFORE the filing, so fyear t -> filing in t+1
    g.index = g.index + 1
    g.index.name = 'filing_year'
    return g


def build() -> pd.DataFrame:
    df = panel().join(national(), how='inner')
    df['panel_hazard'] = df.panel_defaults / df.at_risk
    # implied capture: panel hazard per unit of national filing activity.  Constant if
    # the panel is a stable sample of a universe moving with the national series.
    df['implied_capture'] = df.panel_hazard / df.national_ch11 * 1e6
    return df


if __name__ == '__main__':
    df = build()
    pd.set_option('display.width', 200)
    print('PANEL DEFAULTS vs US COURTS BUSINESS CHAPTER 11 FILINGS')
    print('(panel fiscal year t is aligned to filing year t+1)\n')
    print(df.round(4).to_string())

    cal = df.loc[CALIB_FROM:CALIB_TO]
    k = cal.implied_capture.median()
    lo, hi = cal.implied_capture.quantile([0.10, 0.90])

    print(f'\nimplied capture, {CALIB_FROM}-{CALIB_TO}: median {k:.3f}'
          f'  (10th-90th pct {lo:.3f}-{hi:.3f})')
    print('implied capture, years under suspicion:')
    for y in TEST_YEARS:
        if y in df.index:
            v = df.loc[y, 'implied_capture']
            print(f'   {y}  {v:.3f}   = {v / k:.2f}x the {CALIB_FROM}-{CALIB_TO} median')

    print('\nEXPECTED vs OBSERVED DEFAULTS if capture had been stable')
    tot_e = tot_o = 0
    for y in TEST_YEARS:
        if y not in df.index:
            continue
        r = df.loc[y]
        exp = k * r.national_ch11 * r.at_risk / 1e6
        tot_e += exp; tot_o += r.panel_defaults
        print(f'   {y}  expected {exp:6.1f}   observed {int(r.panel_defaults):4d}'
              f'   shortfall {exp - r.panel_defaults:6.1f}')
    print(f'   {"total":>4s}  expected {tot_e:6.1f}   observed {int(tot_o):4d}'
          f'   observed is {tot_o / tot_e:.1%} of expected')

    rho, p = stats.spearmanr(cal.panel_defaults, cal.national_ch11)
    print(f'\ncaveat on the calculation above: it assumes panel hazard moves with the')
    print(f'national series.  Over {CALIB_FROM}-{CALIB_TO} that assumption is weak '
          f'(Spearman rho {rho:+.3f}, p {p:.3f}) -')
    print('listed-company defaults and all-business Chapter 11 filings are different')
    print('populations.  So treat the shortfall as an order of magnitude, not a count.')

    # The assumption-free version: rank each year in both series.  No proportionality
    # is required - only that a year with heavy national filing activity should not be
    # a year in which a panel of listed companies records its LOWEST default rate.
    print('\nRANK OF EACH YEAR (1 = highest), assumption-free')
    r = pd.DataFrame({
        'national_rank': df.national_ch11.rank(ascending=False).astype(int),
        'panel_hazard_rank': df.panel_hazard.rank(ascending=False).astype(int),
        'panel_hazard_%': (df.panel_hazard * 100).round(2)})
    r['gap'] = r.panel_hazard_rank - r.national_rank
    print(r.to_string())
    print(f'\nmedian panel hazard {df.panel_hazard.median() * 100:.2f}% '
          f'· 2001 {df.loc[2001, "panel_hazard"] * 100:.2f}% '
          f'· 2002 {df.loc[2002, "panel_hazard"] * 100:.2f}%')

    df.to_csv('outputs/external_register.csv')
    print('\nwrote outputs/external_register.csv')
