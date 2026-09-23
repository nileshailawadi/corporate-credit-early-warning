"""
Walk-forward evaluation, and the test of the W2 label-capture finding.

Protocol: for each test year T in 2012..2018, train on fiscal years [start, T-1] whose
outcome window has closed, score the T cross-section, and evaluate on that year alone.
Seven folds instead of one holdout, so estimates rest on several hundred defaults
rather than 119 - the fix W1's wide confidence intervals called for.

The experiment: vary `start`.  If default labels really are incomplete before 2003,
including those years in training should HURT, because the model is being shown deeply
distressed companies carrying a survival label.
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from features import build as features
warnings.filterwarnings('ignore')
RNG = 42
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
              subsample=0.8, colsample_bytree=0.8, random_state=RNG, verbose=-1)


def folds(d, start, first_test=2012, last_test=2018, horizon=1):
    elig = d[f'elig_{horizon}y'] == 1
    for T in range(first_test, last_test + 1):
        tr = elig & (d.fyear >= start) & (d.fyear + horizon - 1 < T)
        te = elig & (d.fyear == T)
        if tr.sum() and te.sum() and d[f'y_{horizon}y'][te].sum() >= 5:
            yield T, tr, te


SEEDS = (0, 1, 2, 3, 4)
# Five seeds, not one.  Fold-level Gini on ~30 defaults carries a seed standard
# deviation of 0.004-0.008, which is the same size as the effects this module is used
# to measure.  Single-seed runs of this comparison flipped sign between two otherwise
# identical feature sets; the seed-averaged answer did not.  Report the average.


def run(d, F, start, horizon=1, seeds=SEEDS):
    rows, pooled_p, pooled_y, pooled_T = [], [], [], []
    y = d[f'y_{horizon}y']
    for T, tr, te in folds(d, start, horizon=horizon):
        yt = y[te].to_numpy()
        k = max(int(0.10 * int(te.sum())), 1)
        ps = []
        for s in seeds:
            m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': s}).fit(F[tr], y[tr])
            ps.append(m.predict_proba(F[te])[:, 1])
        per_seed = [(2 * roc_auc_score(yt, p) - 1, average_precision_score(yt, p),
                     float(yt[np.argsort(-p)[:k]].sum() / yt.sum())) for p in ps]
        g, pr, cap = np.array(per_seed).T
        rows.append(dict(test_year=T, n_train=int(tr.sum()), n_test=int(te.sum()),
                         defaults=int(yt.sum()),
                         gini=g.mean(), gini_sd=g.std(),
                         pr_auc=pr.mean(), top_decile=cap.mean()))
        p_mean = np.mean(ps, axis=0)
        pooled_p.append(p_mean); pooled_y.append(yt); pooled_T.append(np.full(len(p_mean), T))
    r = pd.DataFrame(rows)
    return r, (np.concatenate(pooled_p), np.concatenate(pooled_y), np.concatenate(pooled_T))


def walk(d, F, start, seed, horizon=1):
    """One seed's walk-forward: fold-level Gini and top-decile capture for each test year.

    Reported as a plain mean across folds rather than weighted by default count, so that
    2008 - which carries three times the defaults of a quiet year - cannot dominate the
    headline. Each fold is an independent annual cross-section and counts once.
    """
    y = d[f'y_{horizon}y']
    rows = []
    for T, tr, te in folds(d, start, horizon=horizon):
        yt = y[te].to_numpy()
        m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': seed}).fit(F[tr], y[tr])
        p = m.predict_proba(F[te])[:, 1]
        k = max(int(0.10 * int(te.sum())), 1)
        rows.append(dict(test_year=T, defaults=int(yt.sum()),
                         gini=2 * roc_auc_score(yt, p) - 1,
                         pr_auc=average_precision_score(yt, p),
                         capture=float(yt[np.argsort(-p)[:k]].sum() / yt.sum())))
    r = pd.DataFrame(rows)
    return dict(gini=r.gini.mean(), capture=r.capture.mean(), pr_auc=r.pr_auc.mean(),
                min_gini=r.gini.min(), folds=len(r), defaults=int(r.defaults.sum()))


def summarise(r, pooled, label):
    p, y, T = pooled
    # pooled Gini computed within-year then averaged by default count, so that a year
    # with a high base rate cannot flatter the number
    w = r.defaults / r.defaults.sum()
    return dict(train_from=label,
                folds=len(r), defaults=int(r.defaults.sum()),
                mean_gini=float((r.gini * w).sum()),
                min_gini=float(r.gini.min()),
                mean_top_decile=float((r.top_decile * w).sum()),
                mean_pr_auc=float((r.pr_auc * w).sum()))


def cluster_boot(pa, pb, y, grp, n=600, seed=7):
    rs = np.random.RandomState(seed); firms = np.unique(grp)
    idx = {f: np.where(grp == f)[0] for f in firms}
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[f] for f in rs.choice(firms, len(firms), replace=True)])
        if y[ii].sum() < 10: continue
        out.append(2 * roc_auc_score(y[ii], pb[ii]) - 2 * roc_auc_score(y[ii], pa[ii]))
    return np.array(out)


STARTS = (1999, 2003, 2006)

if __name__ == '__main__':
    raw = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)

    # ---- the start-year experiment, both feature sets, five seeds
    grid = []
    for ext in (False, True):
        F, d = features(raw, extended=ext)
        F, d = F.reset_index(drop=True), d.reset_index(drop=True)
        for start in STARTS:
            for s in SEEDS:
                grid.append(dict(features='extended' if ext else 'core', seed=s,
                                 start=start, **walk(d, F, start, s)))
                print('.', end='', flush=True)
    g = pd.DataFrame(grid)
    g.to_csv('outputs/start_year.csv', index=False)

    pd.set_option('display.width', 200)
    print(f'\n\nWALK-FORWARD, 1-YEAR HORIZON, TEST YEARS 2012-2018')
    print(f'{int(g.defaults.iloc[0])} defaults across {int(g.folds.iloc[0])} annual folds, '
          f'{len(SEEDS)} seeds\n')
    tab = g.groupby(['features', 'start'])[['gini', 'capture']].agg(['mean', 'std'])
    print(tab.round(4).to_string())

    # ---- is dropping 1999-2002 from training a real improvement, or seed noise?
    # Paired within (features, seed): the same learner on the same folds, one difference.
    base = g[g.start == 1999].set_index(['features', 'seed']).gini
    print('\npaired delta vs start=1999')
    for start in STARTS[1:]:
        alt = g[g.start == start].set_index(['features', 'seed']).gini
        delta = (alt - base).dropna()
        print(f'  {start}: {delta.mean():+.4f}   positive in {int((delta > 0).sum())}'
              f'/{len(delta)} paired runs')

    # ---- fold detail for the chosen configuration, and a bootstrap on the same question
    F, d = features(raw)
    F, d = F.reset_index(drop=True), d.reset_index(drop=True)
    r03, (p03, y03, _) = run(d, F, 2003)
    _,   (p99, y99, _) = run(d, F, 1999)
    print('\nFOLD DETAIL, core features, train from 2003 (5 seeds averaged per fold)')
    print(r03.round(4).to_string(index=False))
    r03.to_csv('outputs/walk_forward_folds.csv', index=False)

    # resampling obligors, not rows: a firm's years are not independent observations
    assert (y99 == y03).all(), 'fold alignment mismatch'
    grp = np.concatenate([d.company_name[te].to_numpy() for _, _, te in folds(d, 2003)])
    delta = cluster_boot(p99, p03, y99, grp)
    obs = 2 * roc_auc_score(y03, p03) - 2 * roc_auc_score(y99, p99)
    print(f'\npooled delta-Gini from dropping 1999-2002: {obs:+.4f}'
          f'   95% CI [{np.percentile(delta, 2.5):+.4f}, {np.percentile(delta, 97.5):+.4f}]'
          f'   P(improve) {(delta > 0).mean():.3f}')
    print(f'(cluster bootstrap over obligors, {int(y03.sum())} defaults)')

    (g.groupby(['features', 'start'])[['gini', 'capture', 'pr_auc', 'min_gini']]
       .agg(['mean', 'std']).round(6).to_csv('outputs/walk_forward_summary.csv'))
    print('\nwrote outputs/start_year.csv, outputs/walk_forward_folds.csv, '
          'outputs/walk_forward_summary.csv')
