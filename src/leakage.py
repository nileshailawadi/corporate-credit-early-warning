"""Headline experiment: how much of published performance on this dataset is leakage?"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from audit import load
from features import build as features
warnings.filterwarnings('ignore')
RNG = 42

def fit_eval(Xtr,ytr,Xte,yte,label):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                           random_state=RNG, verbose=-1)
    m.fit(Xtr,ytr)
    p = m.predict_proba(Xte)[:,1]
    k = max(int(0.10*len(p)),1)
    top = np.argsort(-p)[:k]
    return dict(arm=label, n_train=len(ytr), n_test=len(yte), test_pos=int(yte.sum()),
                base_rate=float(yte.mean()),
                accuracy=accuracy_score(yte,(p>0.5).astype(int)),
                roc_auc=roc_auc_score(yte,p), pr_auc=average_precision_score(yte,p),
                gini=2*roc_auc_score(yte,p)-1,
                top_decile_capture=float(yte.values[top].sum()/max(yte.sum(),1)))

if __name__ == '__main__':
    d = load()
    F, d = features(d)
    d['ever_fails'] = (d.status_label=='failed').astype(int)
    # proper point-in-time label: default within 1 year == final observed year of a failed firm
    last = d.groupby('company_name').fyear.transform('max')
    d['default_1y'] = ((d.status_label=='failed') & (d.fyear==last)).astype(int)
    # right-censoring: drop final year of ALIVE firms that exit the panel early (fate unknown)
    censored = (d.status_label=='alive') & (d.fyear==last) & (last<2018)
    cols = F.columns.tolist(); res=[]

    # ARM A — what public notebooks do: company-constant label, random row split
    gss = np.random.RandomState(RNG).rand(len(d))<0.8
    res.append(fit_eval(F[gss],d.ever_fails[gss],F[~gss],d.ever_fails[~gss],
        'A. ever-fails label, RANDOM row split  (the published setup)'))

    # ARM B — same label, but obligors held out
    tr,te = next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=RNG).split(F,d.ever_fails,d.company_name))
    res.append(fit_eval(F.iloc[tr],d.ever_fails.iloc[tr],F.iloc[te],d.ever_fails.iloc[te],
        'B. ever-fails label, OBLIGOR-GROUPED split'))

    # ARM C — grouped AND out-of-time
    trm, tem = d.fyear<=2011, d.fyear>=2015
    seen = set(d.company_name[trm]); tem = tem & ~d.company_name.isin(seen)
    res.append(fit_eval(F[trm],d.ever_fails[trm],F[tem],d.ever_fails[tem],
        'C. ever-fails label, GROUPED + OUT-OF-TIME'))

    # ARM D — the real credit question: PD over the next 12 months
    keep = ~censored
    trm = (d.fyear<=2011)&keep; tem = (d.fyear>=2015)&keep
    res.append(fit_eval(F[trm],d.default_1y[trm],F[tem],d.default_1y[tem],
        'D. 1-YEAR-AHEAD default, GROUPED + OUT-OF-TIME  (the real task)'))

    r = pd.DataFrame(res).set_index('arm')
    pd.set_option('display.width',200)
    print(r[['n_train','n_test','test_pos','base_rate','accuracy','roc_auc','gini','pr_auc','top_decile_capture']]
          .round(4).to_string())
    r.to_csv('outputs/leakage_experiment.csv')
