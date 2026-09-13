"""
W4 - The model ladder.

Each rung has to beat the one below it to survive.  A bank does not adopt a gradient
booster because it is a gradient booster; it adopts one because the scorecard it would
otherwise have to defend to model risk management is measurably worse.

  1. Altman Z''          1968, unfitted, zero parameters. The floor.
  2. WOE logistic        Weight-of-evidence binning + logistic regression. This is what
                         a credit scorecard actually is, and what gets validated.
  3. LightGBM            Gradient boosting on the same features.
  4. Discrete-time       Complementary log-log hazard with a baseline in obligor tenure -
     hazard              the statistically correct framing given right-censoring.

All four are fitted inside each walk-forward fold, on training rows only.  Binning
edges, WOE tables, imputation medians and scalers are fitted on train and applied to
test; nothing crosses the fold boundary.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from evaluate import folds, PARAMS, SEEDS

warnings.filterwarnings('ignore')
N_BINS = 10
SMOOTH = 0.5


# ------------------------------------------------------------------ weight of evidence
class WOE:
    """Quantile binning + weight of evidence. NaN gets its own bin, as it should:
    a missing ratio is information, not something to impute away."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, n_bins=N_BINS):
        self.cols_, self.edges_, self.table_ = list(X.columns), {}, {}
        bad_tot, good_tot = y.sum(), (1 - y).sum()
        for c in self.cols_:
            x = X[c].to_numpy(float)
            e = np.unique(np.nanquantile(x[np.isfinite(x)], np.linspace(0, 1, n_bins + 1)))
            self.edges_[c] = e[1:-1] if len(e) > 2 else np.array([np.nanmedian(x)])
            b = self._bin(x, c)
            w = {}
            for k in range(len(self.edges_[c]) + 2):          # +1 interior, +1 NaN bin
                m = b == k
                bad = y[m].sum() + SMOOTH
                good = (1 - y[m]).sum() + SMOOTH
                w[k] = np.log((bad / (bad_tot + SMOOTH)) / (good / (good_tot + SMOOTH)))
            self.table_[c] = w
        return self

    def _bin(self, x, c):
        b = np.digitize(x, self.edges_[c])
        b = np.where(np.isfinite(x), b, len(self.edges_[c]) + 1)
        return b

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(X), len(self.cols_)))
        for j, c in enumerate(self.cols_):
            b = self._bin(X[c].to_numpy(float), c)
            t = self.table_[c]
            out[:, j] = np.vectorize(lambda k: t.get(k, 0.0))(b)
        return out


# ------------------------------------------------------------------ rungs
def rung_altman(Xtr, ytr, Xte, dtr, dte, seed):
    """No fitting. Lower Z'' is riskier, so the score is its negation."""
    return -Xte['altman_z'].to_numpy(float)


def _prep(Xtr, Xte):
    med = Xtr.median()
    a, b = Xtr.fillna(med), Xte.fillna(med)
    mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).to_numpy(), ((b - mu) / sd).to_numpy()


def rung_woe_logit(Xtr, ytr, Xte, dtr, dte, seed):
    w = WOE().fit(Xtr, ytr)
    m = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')
    m.fit(w.transform(Xtr), ytr)
    return m.predict_proba(w.transform(Xte))[:, 1]


def rung_lightgbm(Xtr, ytr, Xte, dtr, dte, seed):
    m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': seed}).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def rung_hazard(Xtr, ytr, Xte, dtr, dte, seed):
    """Discrete-time hazard: cloglog link with a step baseline in obligor tenure.

    On person-period data a complementary log-log link makes the fitted coefficients
    proportional-hazards, which is the model a survival analyst would write down for
    right-censored panel data.  Falls back to logit if the GLM does not converge.
    """
    import statsmodels.api as sm
    from scipy.linalg import qr
    # WOE-transformed inputs, not raw standardised ratios, so that this rung differs
    # from rung 2 only in the survival framing and the tenure baseline - otherwise the
    # comparison measures functional form (binned vs linear) rather than the thing
    # being tested.  On raw ratios this rung scores 0.551; the gap was the skew of the
    # untransformed inputs, not the hazard formulation.
    w = WOE().fit(Xtr, ytr)
    a, b = w.transform(Xtr), w.transform(Xte)
    ten_tr = np.clip(dtr.tenure.to_numpy(), 0, 9)
    ten_te = np.clip(dte.tenure.to_numpy(), 0, 9)
    Dtr = np.eye(10)[ten_tr]
    Dte = np.eye(10)[ten_te]
    A = np.hstack([a, Dtr])          # no intercept: tenure dummies span it
    B = np.hstack([b, Dte])

    # The feature library contains exact linear dependencies by construction - altman_z
    # IS 6.56*wc_ta + 3.26*re_ta + 6.72*ebit_ta + 1.05*mve_tl.  A GLM on a singular
    # design returns arbitrary coefficients and a useless score (this cost the rung a
    # Gini of 0.06 before it was caught).  Column-pivoted QR picks a maximal independent
    # subset; boosting is indifferent to this, maximum likelihood is not.
    _, r, piv = qr(A, mode='economic', pivoting=True)
    tol = abs(r[0, 0]) * max(A.shape) * np.finfo(float).eps
    keep = np.sort(piv[:int((np.abs(np.diag(r)) > tol).sum())])
    A, B = A[:, keep], B[:, keep]
    for link in (sm.families.links.CLogLog(), sm.families.links.Logit()):
        try:
            g = sm.GLM(ytr, A, family=sm.families.Binomial(link=link))
            r = g.fit(maxiter=200)
            p = r.predict(B)
            if np.isfinite(p).all():
                return p
        except Exception:
            continue
    return rung_woe_logit(Xtr, ytr, Xte, dtr, dte, seed)


# (label, fn, stochastic, outputs_a_probability)
RUNGS = [('1  Altman Z″ (unfitted)', rung_altman, False, False),
         ('2  WOE logistic scorecard', rung_woe_logit, False, True),
         ('3  LightGBM', rung_lightgbm, True, True),
         ('4  Discrete-time hazard', rung_hazard, False, True)]


# ------------------------------------------------------------------ protocol
def metrics(y, p, is_probability):
    """A non-finite score is a firm the model cannot rank; it gets the median score
    rather than being silently dropped, so every model is scored on the same rows."""
    p = np.asarray(p, float)
    bad = ~np.isfinite(p)
    if bad.any():
        p = p.copy(); p[bad] = np.nanmedian(p[~bad]) if (~bad).any() else 0.0
    k = max(int(0.10 * len(p)), 1)
    out = dict(gini=2 * roc_auc_score(y, p) - 1,
               top_decile=float(y[np.argsort(-p)[:k]].sum() / y.sum()),
               pr_auc=average_precision_score(y, p),
               unrankable=float(bad.mean()))
    # Brier only where the output is a probability. Min-max rescaling a raw score and
    # calling the result a Brier score would be meaningless, so Altman gets NaN.
    out['brier'] = brier_score_loss(y, np.clip(p, 0, 1)) if is_probability else np.nan
    return out


def evaluate_ladder(d, F, start=2003, horizon=1, seeds=SEEDS):
    y = d[f'y_{horizon}y']
    out = []
    for name, fn, stochastic, is_prob in RUNGS:
        per_seed = []
        for s in (seeds if stochastic else (SEEDS[0],)):
            rows, wts = [], []
            for T, tr, te in folds(d, start, horizon=horizon):
                p = fn(F[tr], y[tr].to_numpy(), F[te], d[tr], d[te], s)
                rows.append(metrics(y[te].to_numpy(), p, is_prob))
                wts.append(y[te].sum())
            w = np.array(wts) / sum(wts)
            per_seed.append({k: float(np.nansum([r[k] * wi for r, wi in zip(rows, w)]))
                             if np.isfinite([r[k] for r in rows]).any() else np.nan
                             for k in rows[0]})
        agg = pd.DataFrame(per_seed)
        rec = {'model': name}
        for k in agg.columns:
            rec[k] = agg[k].mean()
            rec[k + '_sd'] = agg[k].std() if len(agg) > 1 else 0.0
        out.append(rec)
    return pd.DataFrame(out).set_index('model')


if __name__ == '__main__':
    from features import build
    d = pd.read_parquet('outputs/labelled_panel.parquet').reset_index(drop=True)
    F, d = build(d, extended=True)
    F = F.reset_index(drop=True); d = d.reset_index(drop=True)

    r = evaluate_ladder(d, F)
    show = r[['gini', 'gini_sd', 'top_decile', 'pr_auc', 'brier', 'unrankable']]
    print('MODEL LADDER - walk-forward 2012-2018, train from 2003, 206 defaults')
    print('(LightGBM averaged over 5 seeds; the other three are deterministic)\n')
    print(show.round(4).to_string())

    base = r.loc[RUNGS[0][0], 'gini']
    print('\nlift over the unfitted Altman baseline:')
    for name, _, _, _ in RUNGS[1:]:
        print(f'  {name:30s} {r.loc[name, "gini"] - base:+.4f} Gini')
    r.to_csv('outputs/model_ladder.csv')
    print('\nwrote outputs/model_ladder.csv')
