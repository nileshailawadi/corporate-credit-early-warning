"""Generate the public Kaggle notebook from source, so it cannot drift from the repo.

Every claim in the notebook is either computed in a cell from the attached file, or is a
number this repo's pipeline produced and the notebook says where from. Nothing is asserted
on the strength of having been true in an earlier draft - which is how the first version of
this notebook came to carry two claims that were false.
"""
import json, pathlib

MD, CODE = 'markdown', 'code'
CELLS = []


def md(text):
    CELLS.append({'cell_type': MD, 'metadata': {}, 'source': text.strip()})


def code(text):
    CELLS.append({'cell_type': CODE, 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip('\n')})


md("""
# Label leakage in the American Bankruptcy dataset

This is one of the most heavily worked corporate-default benchmarks in public circulation.
Reported accuracies on it sit between 93% and 100%. Those numbers are real and they mean
almost nothing, because of how the target is built.

Everything below is re-runnable in the cells as they stand. Three things:

1. **The target is a company attribute, not an event.** No company ever changes label, so a
   random row split puts the same obligor on both sides of the test under the same answer.
2. **Default capture is incomplete before 2003.** 2001 and 2002 were among the worst years on
   record for US corporate defaults and are the two quietest years in this panel.
3. **There is a second copy of this dataset in circulation, and it is not interchangeable
   with this one.** Same rows, same companies, same labels — different column order and
   broken magnitudes. There is a one-line check for which copy you are holding.

Then the experiment that matters: one model, one feature set, four evaluation designs.

| | target | split | Gini | defaults caught in the top decile |
|---|---|---|---|---|
| A | ever-fails | random rows *(what most notebooks do)* | 0.654 | 44% |
| B | ever-fails | obligors held out | 0.476 | 32% |
| C | ever-fails | held out + out-of-time | 0.399 | 30% |
| **D** | **default within 12 months** | held out + out-of-time | **0.802** | **72%** |

**Two-fifths of the skill in arm A is the evaluation design.** Asking the real question
instead — *which names deteriorate over the next twelve months* — more than gives it back.

None of this is a criticism of the source paper, whose own method uses obligor-disjoint
splits and an out-of-time test set. It concerns the distributed file and the notebook
ecosystem built on top of it.

*Full pipeline — point-in-time labels, a four-rung model ladder, calibration to a rating
scale, SHAP reason codes and a live watchlist:*
*[github.com/nileshailawadi/corporate-credit-early-warning](https://github.com/nileshailawadi/corporate-credit-early-warning)*
""")

code("""
import numpy as np, pandas as pd, warnings, itertools, glob, os
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)

found = sorted(glob.glob('/kaggle/input/**/*.csv', recursive=True))
found = [f for f in found if 'bankrupt' in os.path.basename(f).lower()] or found
assert found, 'attach utkarshx27/american-companies-bankruptcy-prediction-dataset'
print('reading', found[0])

raw = pd.read_csv(found[0]).rename(columns={'year': 'fyear'})
raw['fyear'] = raw.fyear.astype(int)
print(f'{len(raw):,} firm-years · {raw.company_name.nunique():,} companies · '
      f'{raw.fyear.min()}-{raw.fyear.max()}')
""")

md("""
## First: which copy of this file is this?

Two copies of this panel circulate — this Kaggle dataset, and the authors' own GitHub
release at `sowide/bankruptcy_dataset`. They carry identical rows, identical companies and
identical labels. They do **not** carry their eighteen financial columns in the same order,
and the GitHub copy's magnitudes are broken by powers of 1,000.

So before trusting any `X1..X18` mapping, check it. The sharpest single test is retained
earnings: it is a cumulative figure and goes negative for any firm whose lifetime losses
exceed its lifetime profits. In a panel of listed US companies that is around half of them.
A column that is never negative is not retained earnings.
""")

code("""
MAP = {'X1':'current_assets','X2':'cogs','X3':'dep_amort','X4':'ebitda',
       'X5':'inventory','X6':'net_income','X7':'receivables','X8':'market_value',
       'X9':'net_sales','X10':'total_assets','X11':'lt_debt','X12':'ebit',
       'X13':'gross_profit','X14':'current_liabilities','X15':'retained_earnings',
       'X16':'total_revenue','X17':'total_liabilities','X18':'total_opex'}
d = raw.rename(columns=MAP)

checks = {
    'retained_earnings negative   (expect ~0.5, it is cumulative)': (d.retained_earnings < 0).mean(),
    'net_income        negative   (expect ~0.4)':                   (d.net_income < 0).mean(),
    'total_assets      negative   (must be 0)':                     (d.total_assets < 0).mean(),
    'market_value      negative   (must be 0)':                     (d.market_value < 0).mean(),
    'inventory         negative   (must be 0)':                     (d.inventory < 0).mean(),
}
for k, v in checks.items():
    print(f'{k:62s} {v:.4f}')
""")

md("""
That is the column order the published data dictionary describes, and it holds up: the
signed columns go negative and the unsigned ones never do.

Now the harder test. If the columns are named correctly *and* the magnitudes are intact, the
accounting identities have to close on essentially every row.
""")

code("""
def rel(a, b, tol=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc = np.maximum(np.abs(a), np.abs(b)); sc = np.where(sc < 1e-9, 1.0, sc)
    return np.abs(a - b) / sc < tol

IDS = {'Revenue - OpEx  = EBITDA':      (d.total_revenue - d.total_opex, d.ebitda),
       'EBITDA  - D&A   = EBIT':        (d.ebitda - d.dep_amort,         d.ebit),
       'Revenue - COGS  = GrossProfit': (d.total_revenue - d.cogs,       d.gross_profit)}
for k, (l, r) in IDS.items():
    print(f'{k:30s} holds on {rel(l, r).mean():.4f} of rows')

print(f'\\n{"current assets <= total assets":30s} holds on '
      f'{(d.current_assets <= d.total_assets).mean():.4f} of rows')

s = d.sort_values(['company_name', 'fyear'])
rr = s.groupby('company_name').total_assets.apply(lambda x: (x / x.shift()).dropna())
rr = rr[np.isfinite(rr) & (rr > 0)]
print(f'{"1000x jumps in total assets":30s} {((rr > 100) | (rr < 0.01)).mean():.4f} of year pairs')
print(f'{"median total assets":30s} {d.total_assets.median():,.1f}')
""")

md("""
**This file is clean.** Run the same three cells against the GitHub copy and you get a very
different picture — these are the numbers from `src/mirror_check.py` in the repo:

| | this Kaggle file | `sowide/bankruptcy_dataset` |
|---|---|---|
| columns in data-dictionary order | yes | **no — 16 of 18 differ** |
| `Revenue − OpEx = EBITDA` | 100% of rows | **57%** |
| `EBITDA − D&A = EBIT` | 100% | **56%** |
| current assets ≤ total assets | 100% | **85%** |
| 1,000× jumps in a firm's total assets | 0.1% | **29%** |
| median total assets | 213.2 | 43,266.5 |

Every failing identity in the GitHub copy closes under some power-of-1,000 rescaling of the
individual cells, which is damage rather than a different reporting convention. The column
permutation can be recovered without either data dictionary, from the per-row sign pattern —
multiplying a cell by 1,000 changes neither its sign nor whether it is zero.

A note on how this notebook got here: **its first version claimed the data dictionary was
wrong and the magnitudes were corrupt.** Both claims were made against the GitHub copy and
both are false of this one. The check above is the check that should have been run first, and
it takes two cells. If you are working from a clone rather than from Kaggle, run it.

## Defect 1 — the target is a company attribute, not an event

If the label were an event, a company would be negative until the year it defaults.
""")

code("""
changes = (raw.groupby('company_name').status_label.nunique() > 1).sum()
firms = raw.loc[raw.status_label == 'failed', 'company_name'].nunique()
rows = int((raw.status_label == 'failed').sum())
print(f'companies whose label ever changes : {changes:,} of {raw.company_name.nunique():,}')
print(f'companies that ever fail           : {firms:,}')
print(f'rows labelled failed               : {rows:,}')
print(f'positive rows per failed company   : {rows / firms:.1f}')
""")

md("""
A firm that filed Chapter 11 in 2016 carries `failed` in 1999 too. The modelled question is
not *"will this obligor default next year"* but *"does this company eventually die at some
point over the next two decades"* — and any random split answers it by recognising the
obligor rather than assessing it.

## Defect 2 — default capture is incomplete before 2003

Reconstruct the event. By the dataset's own construction a failed company's last observed
fiscal year is the year before its filing, so that year — and only that year — is the
positive one.
""")

code("""
last = d.groupby('company_name').fyear.transform('max')
d['y_1y'] = ((d.status_label == 'failed') & (d.fyear == last)).astype(int)

hz = d.groupby('fyear').y_1y.agg(defaults='sum', at_risk='size')
hz['hazard_%'] = (hz.defaults / hz.at_risk * 100).round(2)
print(hz.to_string())
""")

md("""
The hazard climbs almost monotonically from 0.06% in 1999 to 1.50% in 2008, then falls. That
is not a credit cycle — it is a capture rate improving as the panel's coverage improves.

US Courts business Chapter 11 filing statistics are a complete administrative count. Rank
every year in both series, assuming no proportionality between them at all:

| filing year | national Ch11 rank | panel hazard rank | panel hazard |
|---|---|---|---|
| 2001 | 3rd of 19 | **19th — last** | 0.13% |
| 2002 | 4th | **18th** | 0.20% |
| 2009 | 1st | 1st | 1.50% |
| *median year* | — | — | 0.91% |

The two heaviest national filing years of the early period are the panel's two quietest. One
honest complication the repo flags rather than resolves: filing year 2010 shows the same
signature inside the supposedly reliable window.

And the few early defaults that *are* labelled do not look like later ones:
""")

code("""
eps = 1e-6
ta = d.total_assets.abs() + eps
d['z'] = (6.56 * (d.current_assets - d.current_liabilities) / ta
          + 3.26 * d.retained_earnings / ta
          + 6.72 * d.ebit / ta
          + 1.05 * d.market_value / (d.total_liabilities.abs() + eps))
era = pd.cut(d.fyear, [1998, 2002, 2006, 2010, 2014, 2018],
             labels=['99-02', '03-06', '07-10', '11-14', '15-18'])
print("median Altman Z-double-prime in the year a company is recorded as defaulting:")
print(d[d.y_1y == 1].groupby(era[d.y_1y == 1], observed=True).z.median().round(2).to_string())
""")

md("""
Companies recorded as defaulting in 1999–2002 are barely distressed on the fundamentals;
those from 2007 onwards are deeply distressed. That is what an incomplete label set looks
like — the distress is visible in the financials of early panel exits, it just never reached
their labels.

The repo trains from 2003 for this reason. It is worth +0.011 Gini while discarding a third
of the training rows, and training from 2006 instead is no better — a shape that supports
the label story rather than a general preference for recent data.

## The experiment: one model, four evaluation designs

Identical features, identical hyper-parameters, test window 2015–2018. Only the target
definition and the split change. Five seeds per arm, because fold-level Gini on this many
defaults carries a seed standard deviation of 0.004–0.008 — the same size as many of the
effects people report from single runs.
""")

code("""
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

# This is the repo's core feature library, minus one sector code the Kaggle file does not
# carry. Ratios, not dollar amounts - a panel spanning four orders of magnitude in firm size
# is not comparable in levels. Trajectories matter as much as levels: deterioration is the
# early-warning signal and the level is only context.
TRAJECTORY = ['leverage', 're_ta', 'ebit_ta', 'current_ratio', 'ebitda_margin']

def features(df):
    df = df.sort_values(['company_name', 'fyear'])
    ta  = df.total_assets.abs() + eps
    rev = df.total_revenue.abs() + eps
    cl  = df.current_liabilities.abs() + eps
    tl  = df.total_liabilities.abs() + eps
    f = pd.DataFrame(index=df.index)

    for c in ['total_assets', 'total_revenue', 'market_value', 'total_liabilities', 'ebitda']:
        f['log_' + c] = np.sign(df[c]) * np.log1p(df[c].abs())      # sign-preserving scale

    f['leverage']       = df.total_liabilities / ta                 # leverage and coverage
    f['lt_debt_ta']     = df.lt_debt / ta
    f['debt_ebitda']    = df.lt_debt / (df.ebitda.abs() + eps)
    f['mve_tl']         = df.market_value / tl

    f['current_ratio']  = df.current_assets / cl                    # liquidity
    f['quick_ratio']    = (df.current_assets - df.inventory) / cl
    f['wc_ta']          = (df.current_assets - df.current_liabilities) / ta

    f['re_ta']          = df.retained_earnings / ta                 # profitability, equity
    f['ebit_ta']        = df.ebit / ta
    f['roa']            = df.net_income / ta
    f['ebitda_margin']  = df.ebitda / rev
    f['net_margin']     = df.net_income / rev
    f['gross_margin']   = df.gross_profit / rev
    f['asset_turnover'] = df.total_revenue / ta

    # Altman Z'' - the emerging-market variant, no sales/TA term, so it travels across sectors
    f['altman_z'] = 6.56 * f.wc_ta + 3.26 * f.re_ta + 6.72 * f.ebit_ta + 1.05 * f.mve_tl

    for c in TRAJECTORY:                                            # 1y and 2y trajectories
        f[c + '_d1'] = f[c] - f[c].groupby(df.company_name).shift(1)
        f[c + '_d2'] = f[c] - f[c].groupby(df.company_name).shift(2)

    f['firm_age'] = df.fyear - df.groupby('company_name').fyear.transform('min')
    return f.replace([np.inf, -np.inf], np.nan), df

F, d = features(d)
F, d = F.reset_index(drop=True), d.reset_index(drop=True)
d['ever_fails'] = (d.status_label == 'failed').astype(int)

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
              subsample=0.8, colsample_bytree=0.8, verbose=-1)

def fit_eval(Xtr, ytr, Xte, yte, seed):
    m = lgb.LGBMClassifier(**PARAMS, random_state=seed).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    yt = np.asarray(yte)
    k = max(int(0.1 * len(p)), 1)
    return dict(gini=2 * roc_auc_score(yt, p) - 1,
                top_decile=yt[np.argsort(-p)[:k]].sum() / yt.sum(),
                accuracy=accuracy_score(yt, (p > 0.5).astype(int)),
                test_defaults=int(yt.sum()))

# Right-censoring matters for arm D: a surviving firm that leaves the panel before 2018 has
# an unknown fate, and labelling that row a survival records an unknown outcome as a good one.
last = d.groupby('company_name').fyear.transform('max')
keep = ~((d.status_label == 'alive') & (d.fyear == last) & (last < 2018))

def four_arms(seed):
    out = {}

    # A - what most public notebooks do: company-constant label, random row split
    m = np.random.RandomState(seed).rand(len(d)) < 0.8
    out['A  ever-fails, random rows'] = fit_eval(F[m], d.ever_fails[m],
                                                 F[~m], d.ever_fails[~m], seed)

    # B - same target, obligors held out
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                  .split(F, d.ever_fails, d.company_name))
    out['B  ever-fails, obligors held out'] = fit_eval(
        F.iloc[tr], d.ever_fails.iloc[tr], F.iloc[te], d.ever_fails.iloc[te], seed)

    # C - obligors held out AND the test years come after the training years
    trm = d.fyear <= 2011
    tem = (d.fyear >= 2015) & ~d.company_name.isin(set(d.company_name[trm]))
    out['C  ever-fails, held out + out-of-time'] = fit_eval(
        F[trm], d.ever_fails[trm], F[tem], d.ever_fails[tem], seed)

    # D - the question a credit team actually asks, on exactly C's split
    out['D  12-month PD, held out + out-of-time'] = fit_eval(
        F[(d.fyear <= 2011) & keep], d.y_1y[(d.fyear <= 2011) & keep],
        F[(d.fyear >= 2015) & keep], d.y_1y[(d.fyear >= 2015) & keep], seed)
    return out
""")

code("""
# The seed drives the split as well as the learner, so arm B's spread is the widest here:
# which obligors land in the test set matters more than which trees get grown.
rows = [dict(arm=a, seed=s, **r) for s in (0, 1, 2, 3, 4) for a, r in four_arms(s).items()]
r = pd.DataFrame(rows)

print(r.groupby('arm')[['gini', 'top_decile', 'accuracy']]
       .agg(['mean', 'std']).round(4).to_string())

A, B, C, D = r.groupby('arm').gini.mean().sort_index().to_numpy()
print(f'\\nholding obligors out          {B - A:+.3f} Gini')
print(f'and moving out of time        {C - B:+.3f} Gini')
print(f'total cost of the design      {C - A:+.3f} Gini   ({1 - C / A:.0%} of arm A)')
print(f'\\nasking the real question instead, on that same hard split: {D:.3f} Gini')
""")

md("""
## Reading the table

**Holding obligors out costs about 18 Gini points** — arm A to arm B. Same model, same
features, same target. The only change is that the test obligors are no longer sitting in the
training set carrying the same answer. That gap is the leakage, measured.

**Moving out of time costs another 8.** Arms B to C. Together, two-fifths of what arm A
reports is the evaluation design rather than the model.

**Arm B's spread is four to five times arm A's**, because the seed here picks the obligors as
well as the trees. Which firms land in the test set matters more than anything about the
learner — another reason a single split is not a result.

**Arm D is the one that matters.** Re-posing the target as a genuine twelve-month default,
on the same hard split as arm C, takes Gini to 0.80 and top-decile capture from 30% to 72%.
The standard framing is therefore both leaky **and** a harder problem than the real one —
*deteriorating now* is a much cleaner signal than *dies eventually*.

**Accuracy rises as the model gets worse.** It climbs from arm A to arm C while every
meaningful measure falls, because the base rate is falling. Any headline accuracy on a
1%-default problem is a statement about the base rate, not about the model.

*(The linked repo runs the same four arms on a feature set that adds an SIC sector code this
file does not carry, and gets 0.697 / 0.485 / 0.357 / 0.813. The gaps are the same story; the
levels move a little with the sector control.)*

## What to do with this dataset instead

- Label the **event**, not the company: positive in the default year only.
- Treat panel exits as **right-censored**. Most surviving companies leave before 2018;
  counting an acquisition or a delisting as a survival labels an unknown outcome as a good
  one.
- Split by **obligor and by time**, and prefer walk-forward to a single holdout — one test
  window here holds 119 defaults, which cannot resolve the differences people report.
- Report **Gini, PR-AUC and capture-at-decile**. Never accuracy.
- Report the **spread over seeds**. Two conclusions in the linked repo reversed sign when
  re-run across five.
- **Check which copy of the file you have** before you map `X1..X18` to anything.

## What the repo does with it

Point-in-time labels with censoring and an exit taxonomy · walk-forward evaluation, one test
year per fold · a four-rung model ladder where a WOE scorecard and a discrete-time hazard
model both match or beat gradient boosting on clean data · isotonic calibration inside each
fold, mapped to a 21-notch master rating scale — reported as a negative result, since the
data supports seven categories and not twenty-one notches · an out-of-time conservatism
overlay that fixes a CCC-C tail no in-sample calibrator can see · SHAP reason codes and a
conceptual-soundness check · a live FY2018 watchlist whose top decile contains 30 of the 36
firms that went on to default.

**[github.com/nileshailawadi/corporate-credit-early-warning](https://github.com/nileshailawadi/corporate-credit-early-warning)**

---

*Dataset: Pellegrino, Lombardo, Adosoglou, Cagnoni, Pardalos & Poggi (2022), "Machine
Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks",
Future Internet 14(8), 244.*
""")

nb = {'cells': CELLS,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                  'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11'}},
      'nbformat': 4, 'nbformat_minor': 5}

out = pathlib.Path(__file__).parent / 'label-leakage-in-the-american-bankruptcy-dataset.ipynb'
out.write_text(json.dumps(nb, indent=1))
print(f'wrote {out} - {len(CELLS)} cells')
