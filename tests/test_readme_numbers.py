"""The README is under test.

Every headline figure quoted in README.md is asserted here against the result table that
produced it. The tables these read are tracked in git, so this runs from a clean clone
without re-running the pipeline.

This exists because the failure mode it guards against actually happened: this repo was
rebuilt on a different copy of the dataset, every number moved, and the prose, the charts
and a published notebook kept quoting the old ones. A number in the README that no table
supports is now a failing test rather than something a reader has to catch.
"""
import pathlib
import numpy as np, pandas as pd, pytest

OUT = pathlib.Path('outputs')
pytestmark = pytest.mark.skipif(not (OUT / 'model_ladder.csv').exists(),
                                reason='result tables missing - run `make all`')

TOL = 5e-4          # the README quotes three decimals


def close(claim, actual, tol=TOL):
    return abs(float(claim) - float(actual)) <= tol


# ------------------------------------------------------------------ the four arms
ARMS = {                      # gini, gini_sd, capture, capture_sd, accuracy
    'A random rows':       (0.697, 0.010, 0.479, 0.006, 0.938),
    'B obligors held out': (0.485, 0.034, 0.331, 0.012, 0.930),
    'C + out-of-time':     (0.357, 0.016, 0.270, 0.018, 0.975),
    'D 12-month PD':       (0.813, 0.005, 0.721, 0.016, 0.990),
}


@pytest.fixture(scope='module')
def arms():
    return pd.read_csv(OUT / 'leakage_experiment.csv').groupby('arm')


@pytest.mark.parametrize('arm', list(ARMS))
def test_arm_matches_readme(arms, arm):
    g, c, a = ARMS[arm][:2], ARMS[arm][2:4], ARMS[arm][4]
    assert close(g[0], arms.gini.mean()[arm]), f'{arm} Gini'
    assert close(g[1], arms.gini.std()[arm]), f'{arm} Gini sd'
    assert close(c[0], arms.top_decile_capture.mean()[arm]), f'{arm} capture'
    assert close(c[1], arms.top_decile_capture.std()[arm]), f'{arm} capture sd'
    assert close(a, arms.accuracy.mean()[arm]), f'{arm} accuracy'


def test_leakage_costs_quoted_in_prose(arms):
    g = arms.gini.mean()
    assert close(0.21, g['A random rows'] - g['B obligors held out'], 0.005)
    assert close(0.13, g['B obligors held out'] - g['C + out-of-time'], 0.005)
    # "roughly half the skill reported in arm A"
    assert 0.45 < 1 - g['C + out-of-time'] / g['A random rows'] < 0.55


def test_accuracy_rises_as_the_model_gets_worse(arms):
    """The point of quoting accuracy at all - it moves the wrong way."""
    a, g = arms.accuracy.mean(), arms.gini.mean()
    assert a['C + out-of-time'] > a['A random rows']
    assert g['C + out-of-time'] < g['A random rows']


# ------------------------------------------------------------------ walk-forward
WF = {1999: (0.819, 0.006, 0.726, 0.009),
      2003: (0.835, 0.004, 0.769, 0.006),
      2006: (0.828, 0.003, 0.756, 0.017)}


@pytest.fixture(scope='module')
def wf():
    return pd.read_csv(OUT / 'start_year.csv')


@pytest.mark.parametrize('start', list(WF))
def test_walk_forward_matches_readme(wf, start):
    c = wf[(wf.features == 'core') & (wf.start == start)]
    gi, gs, ca, cs = WF[start]
    assert close(gi, c.gini.mean()) and close(gs, c.gini.std())
    assert close(ca, c.capture.mean()) and close(cs, c.capture.std())


def test_the_2003_cut_is_worth_what_the_readme_says(wf):
    base = wf[wf.start == 1999].set_index(['features', 'seed']).gini
    d = (wf[wf.start == 2003].set_index(['features', 'seed']).gini - base).dropna()
    assert close(0.012, d.mean())
    assert int((d > 0).sum()) == 9 and len(d) == 10
    # and cutting further to 2006 gives back less, in fewer runs - the shape is the argument
    d6 = (wf[wf.start == 2006].set_index(['features', 'seed']).gini - base).dropna()
    assert close(0.005, d6.mean()) and int((d6 > 0).sum()) == 6
    assert d6.mean() < d.mean()


def test_walk_forward_scale(wf):
    assert int(wf.folds.iloc[0]) == 7 and int(wf.defaults.iloc[0]) == 206


# ------------------------------------------------------------------ the model ladder
LADDER = {'1  Altman Z″ (unfitted)':    (0.629, 0.204, 0.029),
          '2  WOE logistic scorecard':  (0.831, 0.752, 0.135),
          '3  LightGBM':                (0.830, 0.774, 0.269),
          '4  Discrete-time hazard':    (0.843, 0.786, 0.166)}


@pytest.fixture(scope='module')
def ladder():
    return pd.read_csv(OUT / 'model_ladder.csv').set_index('model')


@pytest.mark.parametrize('rung', list(LADDER))
def test_ladder_matches_readme(ladder, rung):
    gi, ca, pr = LADDER[rung]
    assert close(gi, ladder.loc[rung, 'gini']), rung
    assert close(ca, ladder.loc[rung, 'top_decile']), rung
    assert close(pr, ladder.loc[rung, 'pr_auc']), rung


def test_boosting_buys_nothing(ladder):
    """The headline claim of the ladder section, and the one that reversed on clean data."""
    woe = ladder.loc['2  WOE logistic scorecard', 'gini']
    lgbm = ladder.loc['3  LightGBM', 'gini']
    hazard = ladder.loc['4  Discrete-time hazard', 'gini']
    assert abs(lgbm - woe) < 0.005, 'boosting now beats the scorecard - rewrite the section'
    assert hazard > lgbm, 'the hazard model no longer wins - rewrite the section'


def test_altman_ranks_but_misses_the_tail(ladder):
    """Three-quarters of the Gini for zero parameters, and only a fifth of the tail."""
    alt, best = ladder.loc['1  Altman Z″ (unfitted)'], ladder.loc['4  Discrete-time hazard']
    assert 0.73 < alt.gini / best.gini < 0.77
    assert alt.top_decile / best.top_decile < 0.30


# ------------------------------------------------------------------ calibration
@pytest.fixture(scope='module')
def cat():
    return pd.read_csv(OUT / 'master_scale_categories.csv')


def test_portfolio_calibration_matches_readme(cat):
    w = cat.obligor_years
    assert int(w.sum()) == 22755 and int(cat.defaults.sum()) == 206
    assert close(0.865, (cat.mean_pd * w).sum() / w.sum() * 100)
    assert close(0.905, cat.defaults.sum() / w.sum() * 100)


def test_the_notch_scale_is_reported_as_a_negative_result():
    n = pd.read_csv(OUT / 'master_scale_notches.csv')
    assert len(n) == 21
    assert int((n.defaults < 5).sum()) == 8
    cat = pd.read_csv(OUT / 'master_scale_categories.csv')
    ig = cat[cat.category.isin(['AAA', 'AA', 'A'])]
    assert int(ig.obligor_years.sum()) == 6678 and int(ig.defaults.sum()) == 1


def test_overlay_was_needed_in_six_of_seven_years():
    m = pd.read_csv(OUT / 'oot_multipliers.csv').groupby('fyear').raw_ratio.mean()
    assert len(m) == 7 and int((m > 1).sum()) == 6


# ------------------------------------------------------------------ external register
def test_external_register_ranks_match_readme():
    r = pd.read_csv(OUT / 'external_register.csv').set_index('filing_year')
    assert len(r) == 19
    nat = r.national_ch11.rank(ascending=False).astype(int)
    pan = r.panel_hazard.rank(ascending=False).astype(int)
    for yr, n, p in [(2001, 3, 19), (2002, 4, 18), (2009, 1, 1)]:
        assert nat[yr] == n and pan[yr] == p, f'{yr}'
    assert close(0.91, r.panel_hazard.median() * 100, 0.005)
    # the claim in one line: the heaviest early national years are the panel's quietest
    assert pan[2001] > 15 and pan[2002] > 15


# ------------------------------------------------------------------ soundness check
def test_conceptual_soundness_counts_match_readme():
    d = pd.read_csv(OUT / 'direction_check.csv')
    v = d.verdict.value_counts()
    assert len(d) == 17
    assert v.get('agrees', 0) == 7
    assert v.get('non-monotone', 0) == 7
    assert v.get('CONTRADICTS', 0) == 3
    # the one the README names by hand
    row = d[d.driver.str.startswith('Long-term debt')].iloc[0]
    assert row.verdict == 'CONTRADICTS' and row.pd_high < row.pd_low
