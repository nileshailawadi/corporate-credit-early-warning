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


if __name__ == '__main__':
    d = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)
    F, d = features(d)
    F = F.reset_index(drop=True); d = d.reset_index(drop=True)

    print('WALK-FORWARD, 1-YEAR HORIZON, TEST YEARS 2012-2018')
    print('varying the first fiscal year admitted to training\n')
    summ, keep = [], {}
    for start, lab in [(1999, '1999 (all)'), (2003, '2003 (reliable)'), (2006, '2006')]:
        r, pooled = run(d, F, start)
        keep[start] = (r, pooled)
        summ.append(summarise(r, pooled, lab))
        print(f'--- train from {lab}')
        print(r.round(4).to_string(index=False))
        print()

    s = pd.DataFrame(summ).set_index('train_from')
    print('SUMMARY (default-weighted across folds)')
    print(s.round(4).to_string())

    # is 2003 better than 1999, beyond noise?
    (_, (p99, y99, _)), (_, (p03, y03, _)) = keep[1999], keep[2003]
    assert (y99 == y03).all(), 'fold alignment mismatch'
    grp = np.concatenate([d.company_name[te].to_numpy() for _, _, te in folds(d, 2003)])
    delta = cluster_boot(p99, p03, y99, grp)
    obs = 2 * roc_auc_score(y03, p03) - 2 * roc_auc_score(y99, p99)
    print(f'\ndropping 1999-2002 from training:  delta-Gini {obs:+.4f}'
          f'   95% CI [{np.percentile(delta,2.5):+.4f}, {np.percentile(delta,97.5):+.4f}]'
          f'   P(improve) {(delta>0).mean():.3f}')
    print(f'(pooled over {int(y03.sum())} defaults across {len(keep[2003][0])} folds)')

    s.to_csv('outputs/walk_forward_summary.csv')
    keep[2003][0].to_csv('outputs/walk_forward_folds.csv', index=False)
