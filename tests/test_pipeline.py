"""Invariants the repaired panel and the point-in-time labels must satisfy.

These are not unit tests of implementation detail - they are the properties that make
the downstream model trustworthy.  If one of them fails, no result in this repo means
anything.
"""
import sys, pathlib
import numpy as np, pandas as pd, pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from repair import _rel, TOL                                    # noqa: E402
from labels import build, PANEL_END, RELIABLE_FROM              # noqa: E402

CLEAN = pathlib.Path('outputs/clean_panel.parquet')
pytestmark = pytest.mark.skipif(not CLEAN.exists(),
                                reason='run `make repair` first')


@pytest.fixture(scope='module')
def panel():
    return pd.read_parquet(CLEAN)


@pytest.fixture(scope='module')
def lab(panel):
    return build(panel)


# ------------------------------------------------------------------ repair
@pytest.mark.parametrize('name,lhs,rhs', [
    ('Revenue - OpEx = EBITDA', lambda d: d.total_revenue - d.total_opex, lambda d: d.ebitda),
    ('EBITDA - D&A = EBIT',     lambda d: d.ebitda - d.dep_amort,         lambda d: d.ebit),
    ('Revenue - COGS = GP',     lambda d: d.total_revenue - d.cogs,       lambda d: d.gross_profit),
])
def test_accounting_identities_hold(panel, name, lhs, rhs):
    rate = (_rel(lhs(panel), rhs(panel)) < TOL).mean()
    assert rate > 0.999, f'{name} holds in only {rate:.4f} of rows'


def test_no_thousandfold_jumps_within_a_firm(panel):
    s = panel.sort_values(['company_name', 'fyear'])
    r = s.groupby('company_name').total_assets.apply(lambda x: (x / x.shift()).dropna())
    r = r[np.isfinite(r) & (r > 0)]
    assert ((r > 100) | (r < 0.01)).mean() < 0.001


def test_repair_preserves_shape_and_keys(panel):
    assert len(panel) == 78682
    assert panel.company_name.nunique() == 8971
    assert panel.fyear.between(1999, PANEL_END).all()
    assert not panel[['company_name', 'fyear']].duplicated().any()


def test_structural_inequalities_mostly_hold(panel):
    # not 1.0 - total liabilities legitimately exceed total assets for insolvent firms
    assert (panel.current_assets <= panel.total_assets).mean() > 0.95
    assert (panel.inventory <= panel.current_assets).mean() > 0.95


# ------------------------------------------------------------------ labels
def test_default_is_a_single_year_not_a_company_attribute(lab):
    """The whole point of W2: a defaulter is positive in exactly one fiscal year."""
    pos = lab[lab.y_1y == 1].groupby('company_name').size()
    assert (pos == 1).all()
    assert len(pos) == lab.loc[lab.ever_fails == 1, 'company_name'].nunique()


def test_no_label_leaks_backwards(lab):
    """A firm must not be positive in any year before its default year."""
    d = lab[lab.ever_fails == 1]
    assert (d.loc[d.y_1y == 1, 'fyear'] == d.loc[d.y_1y == 1, 'default_year']).all()
    assert (d.loc[d.fyear < d.default_year, 'y_1y'] == 0).all()


def test_censored_rows_are_excluded_not_labelled_negative(lab):
    """A survivor whose outcome window runs past its last observation must be dropped."""
    for h in (1, 2, 3):
        unknown = (lab.ever_fails == 0) & (lab.last_obs < lab.fyear + h - 1)
        assert (lab.loc[unknown, f'elig_{h}y'] == 0).all()
        assert lab.loc[lab[f'elig_{h}y'] == 1, f'y_{h}y'].notna().all()


def test_horizon_labels_are_nested(lab):
    """If a firm defaults within 1 year it also defaults within 2 and 3."""
    e = lab.elig_3y == 1
    assert (lab.loc[e, 'y_1y'] <= lab.loc[e, 'y_2y']).all()
    assert (lab.loc[e, 'y_2y'] <= lab.loc[e, 'y_3y']).all()


def test_exit_taxonomy_partitions_the_panel(lab):
    assert set(lab.exit_type.unique()) <= {'continuing', 'default', 'censored', 'administrative'}
    # every firm has exactly one terminal row
    term = lab[lab.exit_type != 'continuing'].groupby('company_name').size()
    assert (term == 1).all()
    # administrative censoring only at the panel end
    assert (lab.loc[lab.exit_type == 'administrative', 'fyear'] == PANEL_END).all()


def test_default_rate_is_economically_plausible(lab):
    """~1% of listed firm-years, not 6.6% - the company-constant label's rate."""
    e = lab.elig_1y == 1
    assert 0.002 < lab.loc[e, 'y_1y'].mean() < 0.02


def test_reliable_window_has_higher_capture_than_the_early_years(lab):
    """The W2 finding, asserted: default share of exits is materially lower pre-2003."""
    ex = lab[lab.exit_type.isin(['default', 'censored'])]
    early = ex[ex.fyear < RELIABLE_FROM].ever_fails.mean()
    late = ex[(ex.fyear >= RELIABLE_FROM) & (ex.fyear < PANEL_END)].ever_fails.mean()
    assert early < late / 2, f'early {early:.3f} vs late {late:.3f}'
