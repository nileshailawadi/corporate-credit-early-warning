"""Generate the public Kaggle notebook from source, so it cannot drift from the repo."""
import json, pathlib

MD, CODE = 'markdown', 'code'
CELLS = []


def md(text):
    CELLS.append({'cell_type': MD, 'metadata': {}, 'source': text.strip()})


def code(text):
    CELLS.append({'cell_type': CODE, 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip('\n')})


md("""
# Three defects in the American Bankruptcy dataset — and what they do to your model

This dataset is one of the most heavily worked corporate-default benchmarks in public
circulation. Reported accuracies on it sit between 93% and 100%.

This notebook argues those numbers are an artefact, using only checks you can re-run in
the cells below. There are three defects, all verifiable from the file itself:

1. **The target is a company attribute, not an event.** No company ever changes label, so
   a random row split puts the same obligor on both sides of the test.
2. **The published variable dictionary is wrong, and cells carry magnitude errors of exactly
   a factor of 1,000.** The accounting is internally consistent; the file is not. (The repo's
   repair moves 18.8% of cells; the checks here just establish that it needs to happen.)
3. **Default capture is incomplete before 2003.** Confirmed against US Courts filing
   statistics.

Then the experiment that matters: the same model under four evaluation designs. Correcting
the framing takes honest out-of-time discrimination from **Gini 0.39 to 0.73**, and the share
of next year's defaults caught in the top decile from 28% to 58%.

None of this is a criticism of the source paper, whose own method uses obligor-disjoint
splits and an out-of-time test set. It concerns the distributed file and the notebook
ecosystem built on it.

*Full pipeline, tests and the calibrated model:
[github.com/nileshailawadi/corporate-credit-early-warning](https://github.com/nileshailawadi/corporate-credit-early-warning)*
""")

code("""
import numpy as np, pandas as pd, warnings, itertools, os, subprocess
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)

KAGGLE = '/kaggle/input/american-companies-bankruptcy-prediction-dataset/american_bankruptcy.csv'
if os.path.exists(KAGGLE):
    raw = pd.read_csv(KAGGLE)
else:
    if not os.path.exists('bankruptcy_dataset'):
        subprocess.run(['git', 'clone', '--depth', '1', '-q',
                        'https://github.com/sowide/bankruptcy_dataset.git'], check=True)
    raw = pd.read_csv('bankruptcy_dataset/american_bankruptcy_dataset.csv')

raw = raw.rename(columns={'year': 'fyear'})
raw['fyear'] = raw['fyear'].astype(int)
print(f'{len(raw):,} firm-years · {raw.company_name.nunique():,} companies · '
      f'{raw.fyear.min()}-{raw.fyear.max()}')
""")

md("""
## Defect 1 — the target is a company attribute, not an event

If the label were an event, a company would be negative until the year it defaults. It isn't.
""")

code("""
changes = (raw.groupby('company_name').status_label.nunique() > 1).sum()
failed_firms = raw.loc[raw.status_label == 'failed', 'company_name'].nunique()
failed_rows = (raw.status_label == 'failed').sum()

print(f'companies whose label ever changes : {changes:,} of {raw.company_name.nunique():,}')
print(f'companies that ever fail           : {failed_firms:,}')
print(f'rows labelled failed               : {failed_rows:,}')
print(f'positive rows per failed company   : {failed_rows / failed_firms:.1f}')
""")

md("""
A firm that filed Chapter 11 in 2016 is marked `failed` in 1999 too. So the modelled question
is not *"will this obligor default next year"* but *"does this company eventually die at some
point in the next two decades"* — and any random split leaks the answer, because the same
obligor appears on both sides with the same label.
""")

md("""
## Defect 2a — the published variable dictionary is wrong

The dataset authors' own supplementary archive ships the same financials with **named**
columns. Match value multisets between the two and the mapping falls out — and it disagrees
with the published `X1..X18` dictionary.

The quickest check needs no external file at all. Under the published names, two columns are
*identical*, and `retained earnings` is never negative across 78,682 observations.
""")

code("""
X = [f'X{i}' for i in range(1, 19)]
dupes = [(a, b) for a, b in itertools.combinations(X, 2) if np.allclose(raw[a], raw[b])]
print('identical column pairs:', dupes)

PUBLISHED = {'X15': 'retained earnings (published name)', 'X8': 'market value (published name)',
             'X10': 'total assets (published name)'}
for c, label in PUBLISHED.items():
    print(f'{label:42s} share negative = {(raw[c] < 0).mean():.4f}')
""")

md("""
`X14` and `X17` are the same column, and the series the dictionary calls *retained earnings*
is never negative — impossible for a panel of 8,971 listed companies, roughly half of which
carry accumulated deficits.

The mapping recovered by value-matching, which the accounting identities below confirm:
""")

code("""
MAP = {'X1':'current_assets','X2':'total_assets','X3':'cogs','X4':'lt_debt',
       'X5':'dep_amort','X6':'ebit','X7':'ebitda','X8':'gross_profit','X9':'inventory',
       'X10':'current_liabilities','X11':'net_income','X12':'retained_earnings',
       'X13':'receivables','X14':'total_revenue','X15':'market_value',
       'X16':'total_liabilities','X17':'net_sales','X18':'total_opex'}
d = raw.rename(columns=MAP)

print(f"retained_earnings negative : {(d.retained_earnings < 0).mean():.4f}  (plausible)")
print(f"market_value      negative : {(d.market_value < 0).mean():.4f}  (must be 0)")
print(f"total_assets      negative : {(d.total_assets < 0).mean():.4f}  (must be 0)")
""")

md("""
## Defect 2b — magnitudes are corrupted by powers of 1,000

With the corrected names, the accounting identities should hold on every row. They hold on
about half.
""")

code("""
def rel(a, b, tol=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc = np.maximum(np.abs(a), np.abs(b)); sc = np.where(sc < 1e-9, 1.0, sc)
    return np.abs(a - b) / sc < tol

IDS = {'Revenue - OpEx = EBITDA': (d.total_revenue - d.total_opex, d.ebitda),
       'EBITDA - D&A = EBIT':     (d.ebitda - d.dep_amort,         d.ebit),
       'Revenue - COGS = GP':     (d.total_revenue - d.cogs,       d.gross_profit)}
for k, (l, r) in IDS.items():
    print(f'{k:26s} holds on {rel(l, r).mean():.4f} of rows')
""")

md("""
Now rescale individual cells by powers of 1,000 and ask whether *any* combination satisfies
the identity. If the data were merely noisy, most rows would stay broken.
""")

code("""
S = [1.0, 1e3, 1e6, 1e-3]
ebitda, dep, ebit = d.ebitda.values, d.dep_amort.values, d.ebit.values
ok = rel(ebitda - dep, ebit)
base = ok.mean()
for a, b, c in itertools.product(S, repeat=3):
    ok = ok | rel(ebitda * a - dep * b, ebit * c)
print(f'rows satisfying EBITDA - D&A = EBIT as published        : {base:.4f}')
print(f'rows satisfying it under SOME power-of-1000 rescaling   : {ok.mean():.4f}')
""")

md("""
**100%.** The underlying accounting is internally consistent and the file is not: individual
cells carry a wrong power of 1,000. The same signature shows up in the time series — a
company's total assets should not move three orders of magnitude between consecutive years.
""")

code("""
s = d.sort_values(['company_name', 'fyear'])
r = s.groupby('company_name').total_assets.apply(lambda x: (x / x.shift()).dropna())
r = r[np.isfinite(r) & (r > 0)]
print(f'year-on-year total-asset ratios above 100x or below 0.01x: {((r > 100) | (r < 0.01)).mean():.4f}')
""")

md("""
The repo repairs this with dynamic programming over within-firm continuity plus chained
identity solving — all three identities then hold on the full panel, and the 1,000x jumps
fall from 29.3% to 0.02%. Here we carry on with the raw file, because the leakage result
below does not depend on the repair.

## Defect 3 — default capture is incomplete before 2003

Reconstruct the real event: a failed company's last observed fiscal year is, by the dataset's
own construction, the year before its filing.
""")

code("""
last = d.groupby('company_name').fyear.transform('max')
d['default_year'] = np.where(d.status_label == 'failed', last, np.nan)
d['y_1y'] = ((d.status_label == 'failed') & (d.fyear == last)).astype(int)

hz = d.groupby('fyear').y_1y.agg(['sum', 'size'])
hz.columns = ['defaults', 'at_risk']
hz['hazard_%'] = (hz.defaults / hz.at_risk * 100).round(2)
print(hz.to_string())
""")

md("""
The hazard climbs steadily from 0.06% in 1999 to 1.50% in 2008 and then falls back. That is
not a credit cycle. 2001 and 2002 were among the worst years on record for US corporate
defaults, and this panel records ten and seventeen.

US Courts business Chapter 11 filing statistics are a complete administrative count. Rank
every year in both series — no proportionality assumed:

| Filing year | National Ch11 rank | Panel hazard rank | Panel hazard |
|---|---|---|---|
| 2001 | 3rd of 19 | **19th — last** | 0.13% |
| 2002 | 4th | **18th** | 0.20% |
| 2009 | 1st | 1st | 1.50% |
| *median year* | — | — | 0.91% |

The two heaviest national filing years of the early period are the panel's two quietest. And
the few early defaults that *are* labelled look nothing like later ones:
""")

code("""
eps = 1e-6; ta = d.total_assets.abs() + eps
d['z'] = (6.56 * (d.current_assets - d.current_liabilities) / ta
          + 3.26 * d.retained_earnings / ta + 6.72 * d.ebit / ta
          + 1.05 * d.market_value / (d.total_liabilities.abs() + eps))
era = pd.cut(d.fyear, [1998, 2002, 2006, 2010, 2014, 2018],
             labels=['99-02', '03-06', '07-10', '11-14', '15-18'])
defaulters = d[d.y_1y == 1]
print('median Altman Z-double-prime of companies in their default year:')
print(defaulters.groupby(era[d.y_1y == 1], observed=True).z.median().round(2).to_string())
""")

md("""
Median Z'' of **-0.15** for companies recorded as defaulting in 1999-2002, against **-8.00**
from 2007-2010. The later defaulters are deeply distressed; the early ones are barely
distinguishable from healthy firms. Meanwhile the distress *is* visible in the fundamentals of
early panel exits — it just is not in their labels. That is what an incomplete label set looks
like: only the most unambiguous cases got captured, and in the early years not even those.

## The experiment: one model, four evaluation designs

Identical features, identical hyper-parameters, test window 2015-2018. Only the target
definition and the split change.
""")

code("""
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

def features(df):
    df = df.sort_values(['company_name', 'fyear'])
    ta = df.total_assets.abs() + eps; rev = df.total_revenue.abs() + eps
    f = pd.DataFrame(index=df.index)
    for c in ['total_assets', 'total_revenue', 'market_value', 'total_liabilities', 'ebitda']:
        f['log_' + c] = np.sign(df[c]) * np.log1p(df[c].abs())
    f['leverage'] = df.total_liabilities / ta
    f['current_ratio'] = df.current_assets / (df.current_liabilities.abs() + eps)
    f['wc_ta'] = (df.current_assets - df.current_liabilities) / ta
    f['re_ta'] = df.retained_earnings / ta
    f['ebit_ta'] = df.ebit / ta
    f['roa'] = df.net_income / ta
    f['ebitda_margin'] = df.ebitda / rev
    f['mve_tl'] = df.market_value / (df.total_liabilities.abs() + eps)
    f['asset_turnover'] = df.total_revenue / ta
    for c in ['leverage', 're_ta', 'ebit_ta', 'current_ratio']:
        f[c + '_d1'] = f[c] - f[c].groupby(df.company_name).shift(1)
    return f.replace([np.inf, -np.inf], np.nan), df

F, d = features(d)
F, d = F.reset_index(drop=True), d.reset_index(drop=True)
d['ever_fails'] = (d.status_label == 'failed').astype(int)
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
              subsample=0.8, colsample_bytree=0.8, verbose=-1)

def arm(Xtr, ytr, Xte, yte, label, seeds=(0, 1, 2, 3, 4)):
    g, c, a = [], [], []
    for s in seeds:
        m = lgb.LGBMClassifier(**PARAMS, random_state=s).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]; yt = np.asarray(yte)
        k = max(int(.1 * len(p)), 1)
        g.append(2 * roc_auc_score(yt, p) - 1)
        c.append(yt[np.argsort(-p)[:k]].sum() / yt.sum())
        a.append(accuracy_score(yt, (p > .5).astype(int)))
    return dict(arm=label, gini=np.mean(g), gini_sd=np.std(g),
                top_decile=np.mean(c), accuracy=np.mean(a))
""")

md("""
Five seeds per arm, not one. Fold-level Gini on this many defaults carries a seed standard
deviation of 0.004-0.008 — the same size as many of the effects people report from single
runs.
""")

code("""
res = []

# A — what most public notebooks do: company-constant label, random row split
m = np.random.RandomState(0).rand(len(d)) < 0.8
res.append(arm(F[m], d.ever_fails[m], F[~m], d.ever_fails[~m],
               'A  ever-fails, random row split'))

# B — same label, obligors held out
tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=0)
              .split(F, d.ever_fails, d.company_name))
res.append(arm(F.iloc[tr], d.ever_fails.iloc[tr], F.iloc[te], d.ever_fails.iloc[te],
               'B  ever-fails, obligors held out'))

# C — obligors held out AND out of time
trm = d.fyear <= 2011
tem = (d.fyear >= 2015) & ~d.company_name.isin(set(d.company_name[trm]))
res.append(arm(F[trm], d.ever_fails[trm], F[tem], d.ever_fails[tem],
               'C  ever-fails, held out + out-of-time'))

# D — the question a credit team actually asks
censored = (d.status_label == 'alive') & (d.fyear == d.groupby('company_name').fyear.transform('max')) \\
           & (d.fyear < 2018)
keep = ~censored
res.append(arm(F[(d.fyear <= 2011) & keep], d.y_1y[(d.fyear <= 2011) & keep],
               F[(d.fyear >= 2015) & keep], d.y_1y[(d.fyear >= 2015) & keep],
               'D  12-month PD, held out + out-of-time'))

out = pd.DataFrame(res).set_index('arm')
print(out.round(4).to_string())
""")

md("""
## Reading the table

**Holding obligors out costs about 10 Gini points** — arm A to arm B, 0.459 to 0.363. That
drop is the leakage: same model, same features, the only change being that the test obligors
are no longer in training under the same label.

**Arm C does not drop further, and that is worth being straight about.** Moving out of time as
well lands at 0.395, slightly *above* arm B. With a seed spread of 0.021 and far fewer positives
in that window, the honest reading is that B and C are indistinguishable here — not that
out-of-time evaluation is free. The repo's fuller feature set separates them (0.451 to 0.382);
this reduced one does not. Either way the A-to-B gap stands.

**Arm D is the one that matters.** Re-posing the target as a genuine twelve-month default
takes Gini to 0.731 and top-decile capture from 28% to 58%. *Deteriorating now* is a far
cleaner signal than *dies eventually*, so the standard framing is both leaky **and** a harder
problem than the real one.

**Accuracy rises as the model gets worse.** It goes up from arm A to arm C while every
meaningful measure falls, because the base rate drops. Any headline accuracy on a 1%-default
problem is a statement about the base rate, not the model.

## What to do instead

- Label the **event**, not the company: positive in the default year only.
- Treat panel exits as **right-censored**. 5,675 of 8,362 surviving companies leave before
  2018; counting an acquisition as a survival labels an unknown outcome as a good one.
- Split by **obligor and by time**, and prefer walk-forward over a single holdout — one test
  window here holds 119 defaults, which cannot resolve the differences people report.
- Report **Gini, PR-AUC and capture-at-decile**. Never accuracy.
- Report the **spread over seeds**.

The repo carries the full version: the panel repair, point-in-time labels with censoring, a
four-rung model ladder, isotonic calibration with an out-of-time conservatism overlay, a
21-notch master rating scale (reported as a negative result — the data supports seven
categories, not twenty-one notches), SHAP reason codes and 20 passing tests.

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

out = pathlib.Path(__file__).parent / 'three-defects-in-the-american-bankruptcy-dataset.ipynb'
out.write_text(json.dumps(nb, indent=1))
print(f'wrote {out} — {len(CELLS)} cells')
