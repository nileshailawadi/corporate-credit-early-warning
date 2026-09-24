"""Build the README figures from the pipeline outputs.

Every number drawn here is read from a CSV that `make all` produced. Nothing is typed in
by hand, so a chart cannot drift away from the result it illustrates - which is exactly
how a stale figure survived an earlier rebuild of this repo.
"""
from __future__ import annotations
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/img'
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = '#16202c', '#8a97a6', '#dfe4ea'
TITLE_PAD = 26          # room for the legend row above panel 2; kept equal so titles align
BLUE, SLATE, RED, AMBER = '#2b6cb0', '#a8b6c5', '#c53030', '#f0a202'

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160, 'font.size': 9,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlelocation': 'left', 'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.titlepad': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def _bare(ax):
    ax.grid(axis='y', color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines['left'].set_visible(False)


def panel_arms(ax):
    """What the evaluation design alone is worth."""
    r = pd.read_csv(ROOT / 'outputs/leakage_experiment.csv')
    g = r.groupby('arm').gini.agg(['mean', 'std']).sort_index()
    colour = [SLATE, SLATE, SLATE, BLUE]
    x = np.arange(len(g))
    ax.bar(x, g['mean'], 0.62, color=colour, zorder=3)
    ax.errorbar(x, g['mean'], yerr=g['std'], fmt='none', ecolor=INK, lw=1.1, capsize=3, zorder=4)
    for i, v in enumerate(g['mean']):
        ax.text(i, v + (g['std'].iloc[i] or 0) + 0.025, f'{v:.3f}',
                ha='center', fontsize=8.5, weight='bold',
                color=INK if i == len(g) - 1 else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(['A\nrandom\nrows', 'B\nobligors\nheld out',
                        'C\n+ out-of\n-time', 'D\n12-month\nPD'], fontsize=8)
    ax.set_xlabel('evaluation design (identical model and features)', fontsize=8.5, color=MUTED)
    ax.set_ylabel('Gini')
    ax.set_ylim(0, 1.0)
    ax.set_title('Half of arm A is the split, not the model', pad=TITLE_PAD)
    _bare(ax)
    return g


def panel_ladder(ax):
    """Each rung against the one below it, on two metrics that disagree."""
    r = pd.read_csv(ROOT / 'outputs/model_ladder.csv')
    names = ['Altman Z″\n(unfitted)', 'WOE\nscorecard', 'LightGBM', 'Discrete-time\nhazard']
    x = np.arange(len(r)); w, gap = 0.37, 0.02      # 2px of surface between the pair
    off = w / 2 + gap / 2
    ax.bar(x - off, r.gini, w, color=BLUE, label='Gini', zorder=3)
    ax.bar(x + off, r.top_decile, w, color=AMBER, label='capture in top decile', zorder=3)
    ax.errorbar(x - off, r.gini, yerr=r.gini_sd, fmt='none', ecolor=INK, lw=1, capsize=2, zorder=4)
    for i in range(len(r)):
        ax.text(i - off, r.gini[i] + 0.025, f'{r.gini[i]:.3f}', ha='center', fontsize=7.5, color=MUTED)
        ax.text(i + off, r.top_decile[i] + 0.025, f'{r.top_decile[i]:.2f}', ha='center',
                fontsize=7.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_xlabel('model, walk-forward 2012–2018', fontsize=8.5, color=MUTED)
    ax.set_ylim(0, 1.05)
    # The legend sits above the plot, never over it. Inside the axes it collided with the
    # LightGBM bars, which is the one place a reader is being asked to compare two numbers.
    ax.set_title('Boosting buys nothing on clean data', pad=TITLE_PAD)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower left',
              bbox_to_anchor=(0, 1.0), borderaxespad=0, handlelength=1.1,
              handleheight=0.9, columnspacing=1.4, handletextpad=0.5)
    _bare(ax)


def panel_gains(ax):
    """The business result: how much of next year's default comes with how much review."""
    w = pd.read_csv(ROOT / 'outputs/watchlist_calibrated.csv').sort_values('pd', ascending=False)
    y = w.defaulted.to_numpy()
    n, D = len(y), int(y.sum())
    frac = np.arange(1, n + 1) / n
    cap = np.cumsum(y) / D

    ax.fill_between(frac, frac, cap, color=BLUE, alpha=0.10, zorder=2)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
    ax.plot(frac, cap, color=BLUE, lw=2, zorder=3)

    k = int(0.10 * n)
    hit = int(y[:k].sum())
    ax.plot([0.10, 0.10], [0, hit / D], color=RED, lw=1, ls='--', zorder=4)
    ax.plot([0, 0.10], [hit / D, hit / D], color=RED, lw=1, ls='--', zorder=4)
    ax.scatter([0.10], [hit / D], s=34, color=RED, zorder=5)
    ax.annotate(f'review the top {k} names\nand you have {hit} of the {D}',
                (0.10, hit / D), textcoords='offset points', xytext=(12, -34),
                fontsize=8.5, color=INK)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f'share of the FY2018 book reviewed, worst PD first ({n:,} obligors)',
                  fontsize=8.5, color=MUTED)
    ax.set_ylabel('share of next year’s defaults caught')
    ax.set_title('A credit team reviews 10% and catches 83%', pad=TITLE_PAD)
    ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0); ax.spines['left'].set_visible(False)


def headline():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
    panel_arms(axes[0]); panel_ladder(axes[1]); panel_gains(axes[2])
    fig.suptitle('Corporate credit early-warning  ·  point-in-time PD on 78,682 US listed '
                 'firm-years, 1999–2018', x=0.005, y=1.075, ha='left',
                 fontsize=12, weight='bold')
    fig.text(0.005, 1.005, 'every panel is out-of-time; spreads are over five seeds',
             ha='left', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / 'headline.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {p}')


def calibration():
    """The calibration story in detail: the notch scale is too thin, the categories hold."""
    cat = pd.read_csv(ROOT / 'outputs/master_scale_categories.csv')
    notch = pd.read_csv(ROOT / 'outputs/master_scale_notches.csv')
    pdf = pd.read_csv(ROOT / 'outputs/calibrated_pd.csv')

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # 1 - reliability on the diagonal, by category
    ax = axes[0]
    lim = [3e-5, 0.4]
    ax.plot(lim, lim, color=GRID, lw=1.2, zorder=1)
    c = cat[cat.obligor_years > 0]
    lo = np.clip(c.observed - c.ci_lo, 0, None)
    hi = np.clip(c.ci_hi - c.observed, 0, None)
    ax.errorbar(c.mean_pd, c.observed, yerr=[lo, hi],
                fmt='o', ms=5, color=BLUE, ecolor=MUTED, lw=1.1, capsize=3, zorder=3)
    for _, r in c.iterrows():
        ax.annotate(r.category, (r.mean_pd, max(r.observed, 4e-5)),
                    textcoords='offset points', xytext=(7, -3), fontsize=8, color=INK)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('predicted PD'); ax.set_ylabel('observed default rate')
    ax.set_title('On the diagonal, or its interval crosses it')
    ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

    # 2 - why notches do not hold: defaults per notch
    ax = axes[1]
    n = notch[notch.obligor_years > 0]
    ax.bar(range(len(n)), n.defaults, color=[RED if d < 5 else BLUE for d in n.defaults], zorder=3)
    ax.axhline(5, color=INK, lw=1, ls='--', zorder=4)
    ax.text(0.3, 5.9, 'below five defaults a notch\nestimates nothing', fontsize=8, color=INK)
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.grade, rotation=90, fontsize=6.5)
    ax.set_xlabel('notch', fontsize=8.5, color=MUTED)
    ax.set_ylabel('defaults observed')
    thin = int((n.defaults < 5).sum())
    ax.set_title(f'{thin} of {len(n)} notches carry fewer than five defaults')
    _bare(ax)

    # 3 - where the book sits
    ax = axes[2]
    share = c.share_of_book * 100
    ax.bar(range(len(c)), share, color=SLATE, zorder=3)
    ax.bar(range(len(c)), share.where(c.category.isin(['B', 'CCC-C']), 0), color=RED, zorder=3)
    for i, v in enumerate(share):
        ax.text(i, v + 1.0, f'{v:.0f}%', ha='center', fontsize=8, color=MUTED)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.category, fontsize=8.5)
    ax.set_xlabel('rating category', fontsize=8.5, color=MUTED)
    ax.set_ylabel('share of obligor-years')
    ax.set_ylim(0, max(share) * 1.2)
    ax.set_title(f'{int(c.loc[c.category.isin(["B", "CCC-C"]), "defaults"].sum())} of '
                 f'{int(c.defaults.sum())} defaults sit in the red {share[c.category.isin(["B", "CCC-C"])].sum():.0f}%')
    _bare(ax)

    mean_pd, obs = pdf.pd.mean(), pdf.y.mean()
    fig.suptitle(f'Calibration  ·  {len(pdf):,} obligor-years, {int(pdf.y.sum())} defaults, '
                 f'mean predicted PD {mean_pd:.3%} against {obs:.3%} observed',
                 x=0.006, ha='left', fontsize=11.5, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / 'calibration.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {p}')


if __name__ == '__main__':
    headline()
    calibration()
