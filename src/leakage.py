"""How much of the published performance on this dataset is leakage?

Four arms on identical features and an identical learner.  Only the way rows are split
and the way the target is defined change.

  A  company-constant label, random row split      what public notebooks on this dataset do
  B  company-constant label, obligors held out     same target, no obligor on both sides
  C  B, and the test years come after the train    no obligor and no year on both sides
  D  default within the next 12 months, C's split  the question a credit committee asks

A is not a credit model.  609 failed companies contribute 5,220 positive rows, so a
random split puts the same firm on both sides of the test and the model can recognise
it rather than assess it.  B and C strip that away.  D changes the target from "does
this firm eventually die" to "does it default next year" - a harder question that the
model answers far better, because it is a question the financials actually contain.

Everything is averaged over five seeds.  For arms A and B the seed drives the split as
well as the learner, which is why arm B's spread is the widest here: which obligors land
in the test set matters more than which trees get grown.
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from data import load
from features import build as features
warnings.filterwarnings('ignore')

SEEDS = (0, 1, 2, 3, 4)
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
              subsample=0.8, colsample_bytree=0.8, verbose=-1)


def fit_eval(Xtr, ytr, Xte, yte, seed):
    m = lgb.LGBMClassifier(**PARAMS, random_state=seed).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    yte = np.asarray(yte)
    k = max(int(0.10 * len(p)), 1)
    top = np.argsort(-p)[:k]
    return dict(n_train=len(ytr), n_test=len(yte), test_pos=int(yte.sum()),
                base_rate=float(yte.mean()),
                accuracy=accuracy_score(yte, (p > 0.5).astype(int)),
                roc_auc=roc_auc_score(yte, p),
                gini=2 * roc_auc_score(yte, p) - 1,
                pr_auc=average_precision_score(yte, p),
                top_decile_capture=float(yte[top].sum() / max(yte.sum(), 1)))


def arms(d, F, seed):
    """One draw of all four arms under a single seed."""
    rs = np.random.RandomState(seed)
    out = {}

    # A - company-constant label, random row split
    m = rs.rand(len(d)) < 0.8
    out['A random rows'] = fit_eval(F[m], d.ever_fails[m], F[~m], d.ever_fails[~m], seed)

    # B - same label, obligors held out
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                  .split(F, d.ever_fails, d.company_name))
    out['B obligors held out'] = fit_eval(F.iloc[tr], d.ever_fails.iloc[tr],
                                         F.iloc[te], d.ever_fails.iloc[te], seed)

    # C - obligors held out AND the test years come after the training years
    trm = d.fyear <= 2011
    tem = (d.fyear >= 2015) & ~d.company_name.isin(set(d.company_name[trm]))
    out['C + out-of-time'] = fit_eval(F[trm], d.ever_fails[trm], F[tem], d.ever_fails[tem], seed)

    # D - the real credit question, on C's split
    keep = ~d.censored_row
    trm, tem = (d.fyear <= 2011) & keep, (d.fyear >= 2015) & keep
    out['D 12-month PD'] = fit_eval(F[trm], d.default_1y[trm], F[tem], d.default_1y[tem], seed)
    return out


if __name__ == '__main__':
    d = load()
    F, d = features(d)
    F, d = F.reset_index(drop=True), d.reset_index(drop=True)

    d['ever_fails'] = (d.status_label == 'failed').astype(int)
    last = d.groupby('company_name').fyear.transform('max')
    # point-in-time target: a failed firm is positive only in its final observed year
    d['default_1y'] = ((d.status_label == 'failed') & (d.fyear == last)).astype(int)
    # right-censoring: a survivor's last year before the panel ends has an unknown fate
    d['censored_row'] = (d.status_label == 'alive') & (d.fyear == last) & (last < 2018)

    rows = []
    for s in SEEDS:
        for arm, r in arms(d, F, s).items():
            rows.append(dict(arm=arm, seed=s, **r))
            print('.', end='', flush=True)
    r = pd.DataFrame(rows)

    agg = (r.groupby('arm')[['gini', 'top_decile_capture', 'pr_auc', 'accuracy']]
             .agg(['mean', 'std']))
    counts = r.groupby('arm')[['n_train', 'n_test', 'test_pos', 'base_rate']].mean()

    pd.set_option('display.width', 200)
    print(f'\n\nLEAKAGE ARMS, mean +/- sd over {len(SEEDS)} seeds\n')
    print(agg.round(4).to_string())
    print('\nsplit sizes (mean over seeds)')
    print(counts.round(4).to_string())
    G = agg[('gini', 'mean')]
    print(f'\nwhat the evaluation design alone costs, target held fixed:')
    print(f'  holding obligors out          {G["B obligors held out"] - G["A random rows"]:+.4f} Gini')
    print(f'  and moving out of time        {G["C + out-of-time"] - G["B obligors held out"]:+.4f} Gini')
    print(f'  total                         {G["C + out-of-time"] - G["A random rows"]:+.4f} Gini'
          f'   ({1 - G["C + out-of-time"] / G["A random rows"]:.0%} of arm A)')
    print(f'\nre-posing the target as a real 12-month PD, on that same hard split:'
          f' {G["D 12-month PD"]:.4f} Gini.')
    print('Arm A is not a better model than arm D. It is an easier, useless question.')

    r.to_csv('outputs/leakage_experiment.csv', index=False)
    agg.round(6).to_csv('outputs/leakage_summary.csv')
    print('\nwrote outputs/leakage_experiment.csv, outputs/leakage_summary.csv')
