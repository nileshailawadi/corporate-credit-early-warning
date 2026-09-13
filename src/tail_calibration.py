"""
Fixing the CCC-C miscalibration.

W5 left the worst grade under-predicted: 9.4% against 12.8% observed, with the Wilson
interval excluding the prediction.  Under-stating risk in the grade a watchlist is
built from is the wrong direction to be wrong in, so this module first diagnoses the
cause and then compares three calibrators on identical base scores.

Two candidate causes, and they need different fixes:

  (a) Isotonic tail compression.  Isotonic regression is a step function.  Its top
      block averages every training score above the last change point, so the most
      extreme predictions get pulled down toward the block mean.  Fixable with a
      calibrator that extrapolates instead of saturating.

  (b) Out-of-time drift.  If tail default rates are simply higher in the test years
      than in the training years, no calibrator fitted on the training period can know
      that, and the fix is recalibration frequency, not calibrator choice.

The diagnosis is per-test-year: (a) is roughly constant across years, (b) is not.

Efficiency note: the base model is fitted once per (fold, seed) and its out-of-fold
training scores are shared by all three calibrators, rather than refitting the model
inside each CalibratedClassifierCV.  Same statistics, a third of the compute.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from evaluate import folds, PARAMS, SEEDS
from calibrate import grade, CATEGORY, wilson, scale_table

warnings.filterwarnings('ignore')
EPS = 1e-7


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Isotonic:
    name = 'isotonic'

    def fit(self, s, y):
        self.m = IsotonicRegression(out_of_bounds='clip').fit(s, y)
        return self

    def predict(self, s):
        return np.clip(self.m.predict(s), EPS, 1 - EPS)


class Platt:
    name = 'platt (logit)'

    def fit(self, s, y):
        self.m = LogisticRegression(max_iter=1000).fit(_logit(s).reshape(-1, 1), y)
        return self

    def predict(self, s):
        return self.m.predict_proba(_logit(s).reshape(-1, 1))[:, 1]


class IsotonicTail:
    """Isotonic through the bulk, logistic-in-logit above a quantile threshold.

    Isotonic is the better calibrator where data is dense and the worse one where it
    runs out, because a step function cannot extrapolate.  Above `tail_q` this hands
    over to a two-parameter logistic fitted on the tail alone, which can.  The join is
    made monotone by construction.
    """
    name = 'isotonic + logit tail'

    def __init__(self, tail_q=0.90):
        self.tail_q = tail_q

    def fit(self, s, y):
        self.iso = IsotonicRegression(out_of_bounds='clip').fit(s, y)
        self.thr = float(np.quantile(s, self.tail_q))
        m = s >= self.thr
        if y[m].sum() >= 10 and len(np.unique(y[m])) == 2:
            self.lr = LogisticRegression(max_iter=1000).fit(_logit(s[m]).reshape(-1, 1), y[m])
        else:
            self.lr = None
        return self

    def predict(self, s):
        p = np.clip(self.iso.predict(s), EPS, 1 - EPS)
        if self.lr is not None:
            m = s >= self.thr
            if m.any():
                p[m] = self.lr.predict_proba(_logit(s[m]).reshape(-1, 1))[:, 1]
        # enforce monotonicity in s across the join
        o = np.argsort(s)
        p[o] = np.maximum.accumulate(p[o])
        return p


class IsotonicOverlay:
    """Isotonic, plus a margin of conservatism on the lowest grades.

    Three calibrator families all under-predict CCC-C by 25-35%, consistently across six
    of seven test years.  That rules out the functional form: a monotone remap cannot add
    resolution the base scores do not have, and the calibration sample holds too few tail
    defaults to pin the top of the curve.

    What supervisors expect in exactly this situation is not a cleverer estimator but a
    documented margin of conservatism where the estimate is known to be uncertain.  The
    overlay is a single multiplier on PDs above the CCC+ boundary, estimated from the
    observed/predicted ratio in that band ON TRAINING DATA ONLY, and capped.  If the gap
    is structural it transfers out of time; if it is noise it will not, and this arm
    will show that.
    """
    name = 'isotonic + conservatism overlay'
    BAND = 0.048          # CCC+ lower bound on the master scale
    CAP = 3.0

    def fit(self, s, y):
        self.iso = IsotonicRegression(out_of_bounds='clip').fit(s, y)
        p = np.clip(self.iso.predict(s), EPS, 1 - EPS)
        m = p >= self.BAND
        if m.sum() >= 50 and p[m].mean() > 0:
            self.mult = float(np.clip(y[m].mean() / p[m].mean(), 1.0, self.CAP))
        else:
            self.mult = 1.0
        return self

    def predict(self, s):
        p = np.clip(self.iso.predict(s), EPS, 1 - EPS)
        m = p >= self.BAND
        p[m] = np.clip(p[m] * self.mult, 0, 1 - EPS)
        return p


CALIBRATORS = [Isotonic, Platt, IsotonicTail, IsotonicOverlay]


def calibrated_fold(Xtr, ytr, Xte, seed, n_splits=3):
    """One (model, calibrator) pair PER cv split, averaged - i.e. exactly what
    CalibratedClassifierCV does, with the three calibrators sharing the base fits.

    Getting this wrong is easy and expensive.  A faster-looking arrangement - pool the
    out-of-fold scores, fit ONE calibrator on them, apply it to a model refitted on all
    of train - gives a calibrator trained on scores from a model fitted on n(k-1)/k rows
    and applies it to systematically sharper full-train scores.  That version pushed
    portfolio PD to 1.69x the observed default rate while every component looked
    individually correct.  The `isotonic` arm below exists as a CONTROL: it must
    reproduce the W5 result (portfolio ratio near 0.97, CCC-C near 9.4% predicted) or
    the harness is wrong and none of the comparison means anything.
    """
    parts = {C.name: [] for C in CALIBRATORS}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': seed}).fit(Xtr.iloc[a], ytr[a])
        s_hold = m.predict_proba(Xtr.iloc[b])[:, 1]
        s_test = m.predict_proba(Xte)[:, 1]
        for C in CALIBRATORS:
            parts[C.name].append(C().fit(s_hold, ytr[b]).predict(s_test.copy()))
    return {k: np.mean(v, axis=0) for k, v in parts.items()}


def run(d, F, start=2003, horizon=1, seeds=SEEDS):
    y = d[f'y_{horizon}y']
    rows = []
    for T, tr, te in folds(d, start, horizon=horizon):
        ytr = y[tr].to_numpy()
        preds = {c.name: [] for c in CALIBRATORS}
        for s in seeds:
            for name, p in calibrated_fold(F[tr], ytr, F[te], s).items():
                preds[name].append(p)
        for name, ps in preds.items():
            rows.append(pd.DataFrame({
                'calibrator': name, 'fyear': T,
                'company': d.company_name[te].to_numpy(),
                'pd': np.mean(ps, axis=0), 'y': y[te].to_numpy()}))
    r = pd.concat(rows, ignore_index=True)
    r['grade'] = grade(r.pd.to_numpy())
    r['category'] = r.grade.map(CATEGORY)
    return r


def tail_report(r):
    out = []
    for name, g in r.groupby('calibrator'):
        t = g[g.category == 'CCC-C']
        lo, hi = wilson(t.y.sum(), len(t))
        out.append(dict(calibrator=name,
                        portfolio_pd=g.pd.mean(), portfolio_obs=g.y.mean(),
                        portfolio_ratio=g.pd.mean() / g.y.mean(),
                        gini=2 * roc_auc_score(g.y, g.pd) - 1,
                        brier=brier_score_loss(g.y, g.pd),
                        cccc_n=len(t), cccc_pd=t.pd.mean(), cccc_obs=t.y.mean(),
                        cccc_ci_lo=lo, cccc_ci_hi=hi,
                        cccc_covered=lo <= t.pd.mean() <= hi))
    return pd.DataFrame(out).set_index('calibrator')


if __name__ == '__main__':
    from features import build
    d = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)
    F, d = build(d, extended=True)
    F = F.reset_index(drop=True); d = d.reset_index(drop=True)

    r = run(d, F)
    r.to_csv('outputs/tail_calibration.csv', index=False)
    pd.set_option('display.width', 240)

    # CONTROL: the isotonic arm must reproduce W5 before anything else is believable.
    ctl = r[r.calibrator == 'isotonic']
    ratio = ctl.pd.mean() / ctl.y.mean()
    print(f'CONTROL - isotonic arm portfolio PD / observed = {ratio:.3f}'
          f'   (W5 via CalibratedClassifierCV: 0.967)')
    if not 0.90 <= ratio <= 1.05:
        print('  *** harness does not reproduce W5 - the comparison below is NOT valid ***')
    print()

    print('DIAGNOSIS - is CCC-C under-prediction constant across test years (tail')
    print('compression) or concentrated in particular years (out-of-time drift)?\n')
    iso = r[r.calibrator == 'isotonic']
    t = iso[iso.category == 'CCC-C'].groupby('fyear').agg(
        n=('y', 'size'), defaults=('y', 'sum'), predicted=('pd', 'mean'), observed=('y', 'mean'))
    t['ratio'] = t.observed / t.predicted
    print(t.round(4).to_string())
    print(f'\nratio across years: min {t.ratio.min():.2f}  median {t.ratio.median():.2f}'
          f'  max {t.ratio.max():.2f}  sd {t.ratio.std():.2f}')

    print('\n\nCALIBRATOR COMPARISON')
    print(tail_report(r).round(4).to_string())

    print('\n\nPER-CATEGORY, BEST CALIBRATOR BY CCC-C COVERAGE')
    rep = tail_report(r)
    best = rep[rep.cccc_covered].index.tolist() or [rep.cccc_ratio.idxmax()
                                                    if 'cccc_ratio' in rep else rep.index[0]]
    for name in dict.fromkeys(best):
        print(f'\n--- {name}')
        g = r[r.calibrator == name]
        tc = scale_table(g, 'category', ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC-C'])
        print(tc.assign(mean_pd=lambda x: (x.mean_pd * 100).round(3),
                        observed=lambda x: (x.observed * 100).round(3),
                        ci_lo=lambda x: (x.ci_lo * 100).round(3),
                        ci_hi=lambda x: (x.ci_hi * 100).round(3),
                        share_of_book=lambda x: (x.share_of_book * 100).round(1)).to_string())
    print('\nwrote outputs/tail_calibration.csv')
