# Corporate Credit Early-Warning System

A point-in-time probability-of-default model for US listed corporates, built to answer the
question a credit committee actually asks — *which names in this book deteriorate over the next
twelve months?* — rather than the question the public benchmark accidentally poses.

**78,682 firm-years · 8,971 companies · 609 defaults · 1999–2018**

**[→ Open the live watchlist](https://nileshailawadi.github.io/corporate-credit-early-warning/)** — the 2018 book ranked by calibrated PD, with rating grades and reason codes on every name.

**[→ Read the notebook on Kaggle](https://www.kaggle.com/code/nileshailawadi/label-leakage-in-the-american-bankruptcy-dataset)** — the leakage experiment on its own, runnable against the dataset in place.

![headline](docs/img/headline.png)

---

## Which copy of the dataset you use decides your results

Two copies of the American Bankruptcy panel are in circulation. They contain the same 78,682
rows, the same 8,971 companies, the same 609 defaults and the same year distribution — every
count matches exactly. They are **not** interchangeable.

| | [Kaggle](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) | [GitHub](https://github.com/sowide/bankruptcy_dataset) |
|---|---|---|
| Column order | matches its data dictionary | **16 of 18 positions differ** |
| `Revenue − OpEx = EBITDA` | 99.99% of rows | **56.7%** |
| `EBITDA − D&A = EBIT` | 100% | **56.0%** |
| Current assets ≤ total assets | 100% | **85.2%** |
| 1,000× jumps in a firm's total assets | 0.1% | **29.3%** |
| Median total assets | 213.2 | 43,266.5 |

The GitHub copy — the authors' own release — has its eighteen financial columns in a different
order from the Kaggle mirror, and individual cells have lost their magnitude by powers of 1,000.
Every failing accounting identity there resolves under 1000^k rescaling, which is damage rather
than a different vintage. **Anyone who clones the repo instead of downloading from Kaggle gets
scrambled columns and broken magnitudes, with nothing to signal it.**

The permutation is recovered in `mirror_check.py` without reference to either data dictionary:
multiplying a cell by 1,000 changes neither its sign nor whether it is zero, so the exact per-row
sign pattern identifies thirteen columns across the two copies, and matching values up to a power
of 1,000 resolves the remaining five at a score of 1.000.

This project was originally built on the GitHub copy and reported its defects as defects of the
dataset. They are not. `src/mirror_check.py` establishes the difference, `src/data.py` loads the
Kaggle copy and refuses to silently substitute the other, and everything below is measured on the
clean file.

---

## What is actually wrong with the dataset

Two things, and both survive on the clean copy because both live in the label rather than the
financials.

### 1. The target is a company attribute, not an event

Zero of 8,971 companies change label across their panel history. A firm that filed Chapter 11 in
2016 is marked `failed` in 1999 too — 609 failed companies generate 5,220 positive rows, an
average of 8.6 each. Any random row split therefore puts the same obligor on both sides of the
test, and the modelled question silently becomes *"does this firm eventually die within two
decades"*.

### 2. Default capture is incomplete before 2003

Over 1999–2002, 1,670 companies leave the panel and only 37 of those exits are labelled defaults.
The annual default rate climbs almost monotonically from 0.06% in 1999 to 1.50% in 2008 and then
falls back — which is not a credit cycle.

**Confirmed against an external register.** US Courts business Chapter 11 filing statistics are a
complete administrative count. Ranking every year in both series, with no proportionality assumed:

| Filing year | National Ch11 rank | Panel hazard rank | Panel hazard |
|---|---|---|---|
| 2001 | 3rd of 19 | **19th — last** | 0.13% |
| 2002 | 4th | **18th** | 0.20% |
| 2009 | 1st | 1st | 1.50% |
| *median year* | — | — | 0.91% |

The two heaviest national filing years of the early period are the panel's two quietest. One
honest complication: filing year 2010 shows the same signature inside the supposedly reliable
period, plausibly population mix, and it is flagged rather than resolved.

---

## What correcting the framing is worth

Identical model, identical features, four evaluation designs, test window 2015–2018.
Mean ± sd over 5 seeds, on the clean file:

| Arm | Target | Split | Gini | Top-decile capture | Accuracy |
|---|---|---|---|---|---|
| A | ever-fails | random rows *(the published setup)* | 0.697 ± 0.010 | 0.479 ± 0.006 | 93.8% |
| B | ever-fails | obligors held out | 0.485 ± 0.034 | 0.331 ± 0.012 | 93.0% |
| C | ever-fails | held out + out-of-time | 0.357 ± 0.016 | 0.270 ± 0.018 | 97.5% |
| **D** | **12-month PD** | **held out + out-of-time** | **0.813 ± 0.005** | **0.721 ± 0.016** | 99.0% |

Holding obligors out costs 21 Gini points. Also moving out of time costs another 13. Together,
**roughly half the skill reported in arm A is an artefact of the evaluation design.** Re-posing
the target as a genuine twelve-month default then more than doubles it back, because
*deteriorating now* is a far cleaner signal than *dies eventually*.

Accuracy *rises* as the model gets worse, because the base rate falls. It is not reported anywhere
else in this repo.

Walk-forward — expanding-window train, one test year per fold, 2012–2018, 206 defaults, 5 seeds:

| Train from | Gini | Top-decile capture |
|---|---|---|
| 1999 (all) | 0.819 ± 0.006 | 0.726 ± 0.009 |
| **2003 (reliable labels)** | **0.835 ± 0.004** | **0.769 ± 0.006** |
| 2006 | 0.828 ± 0.003 | 0.756 ± 0.017 |

Training from 2003 beats training from 1999 by +0.012 Gini, positive in 9 of 10 paired runs, while
*discarding a third of the training rows*. Cutting further to 2006 gives back only +0.005 and in
6 runs of 10. The peak in the middle supports the label-capture story; a general preference for
recent data would keep improving.

---

## The model ladder, and why the simple models win

Each rung has to beat the one below it. Walk-forward 2012–2018, train from 2003, 206 defaults:

| | Gini | Top-decile capture | PR-AUC |
|---|---|---|---|
| 1 Altman Z″, unfitted | 0.629 | 0.204 | 0.029 |
| 2 WOE logistic scorecard | 0.831 | 0.752 | 0.135 |
| 3 LightGBM | 0.830 ± 0.002 | 0.774 | 0.269 |
| **4 Discrete-time hazard** (cloglog + tenure baseline) | **0.843** | **0.786** | 0.166 |

**Gradient boosting buys nothing here.** The WOE scorecard matches it on Gini (0.831 against
0.830), and the discrete-time hazard model beats both. On the damaged copy of the data boosting
appeared to win by four Gini points; on clean data that advantage disappears entirely. Complexity
was being paid to absorb corruption.

That is the useful finding for anyone building this for a bank: the two models that would clear
model validation without argument are also the two best ones.

**The 1968 formula gets three-quarters of the Gini for zero parameters — and only 20% of the
tail.** Altman Z″ ranks the population respectably and is nearly useless in the decile a watchlist
operates in. Gini alone hides that; capture-at-decile exposes it.

---

## Calibration and the rating scale

A score that ranks is not a PD. Isotonic regression fitted **inside each fold** on cross-validated
training predictions — calibrating on in-sample scores is the usual way this goes wrong.

Across 22,755 obligor-years and 206 defaults: **mean predicted PD 0.865% against an observed
default rate 0.905%**. The Murphy decomposition puts the miscalibration term at 0.000018 of a
Brier score of 0.0077, about 0.2%.

![calibration](docs/img/calibration.png)

### 21 notches, attempted

The scale is an **internal master scale wearing agency-style labels** — 21 grades, geometric PD
bands at a ratio of 1.5 per notch. It is not a claim that this model's BBB is S&P's BBB.

Attempted at notch level, the data does not support it: 8 of the 21 notches carry fewer than five
defaults, and the AAA–A end of the scale carries a single default across 6,678 obligor-years — one
event cannot estimate three grades. Rolled up to seven rating categories it holds, and the observed
default rate is monotone across all seven.

### The tail, and how it was fixed

CCC-C initially under-predicted: 9.9% predicted against 13.8% observed. Three calibrator families
all failed to fix it. The informative arm was a conservatism overlay estimated **in-sample** — it
clipped to 1.0, meaning the gap does not exist in training data at all. Out of time it does, by a
median of 1.27× across six of seven test years.

So it is neither isotonic tail compression (which would show in-sample) nor concentrated drift
(which would show in one or two years). It is an out-of-time generalisation gap, and no calibrator
fitted on in-sample data can see it. Estimating the overlay on a **held-out prior year** fixes it:

| Calibrator | Portfolio ratio | Gini | CCC-C predicted | observed | covered |
|---|---|---|---|---|---|
| isotonic | 0.953 | 0.8457 | 10.56% | 13.63% | no |
| isotonic + logit tail | 0.942 | 0.8461 | 10.42% | 14.30% | no |
| platt | 0.964 | 0.8445 | 11.73% | 15.66% | no |
| in-sample overlay | 0.953 | 0.8457 | 10.56% | 13.63% | no |
| **isotonic + out-of-time overlay** | 1.055 | 0.8459 | **12.10%** | 13.41% | **yes** |

All seven categories now contain their own predicted PD, at the cost of mild overall conservatism
(portfolio ratio 1.055) and no discrimination. The overlay is floored at 1.0, so it can only ever
make the model more conservative.

The harness carries a **control arm that must reproduce a known result** before any comparison is
believed — it caught an earlier calibrator/base-model mismatch that had inflated portfolio PD to
1.69× observed while every component looked individually correct.

---

## The watchlist

FY2018 scored with the full production path: 2,723 obligors, of which 36 subsequently defaulted.
The top decile — 272 names a credit team could realistically review — contains **30 of the 36**.
Gini on that cross-section is 0.857.

Each name carries its three strongest drivers from SHAP attributions, rendered as analyst lines
(*"market value / total liabilities 0.00× — 0th percentile of book"*).

A conceptual-soundness check runs one-way partial dependence against credit intuition on 17
drivers: **7 agree, 7 are non-monotone, 3 contradict it.** Long-term debt to EBITDA reads as
*lower* PD as it rises, plausibly because access to term debt is itself a credit signal while
distressed small caps carry none and fail on cash burn. One-way PDP holds correlated features at
observed values, so these are questions to answer rather than verdicts — but they are the
questions a model owner has to be able to answer in a validation meeting, and the check surfaces
them by name rather than leaving them to be discovered there.

---

## Honest limitations

- **This project was rebuilt.** The first version used the GitHub copy and reported its damage as
  defects of the dataset. That was wrong, and the numbers above replace the originals. The commit
  history keeps the mistake visible rather than tidying it away.
- **The 2003 cut is a judgement call** worth +0.012 Gini. What carries it is the external register
  and the shape across three training windows, not the delta alone.
- **Exits are pooled as one censoring event.** Acquisition, delisting and loss of coverage cannot
  be distinguished. Treating exits as survivals — the alternative — is strictly worse.
- **The out-of-time overlay rests on one held-out year per fold**, and is floored rather than free
  to adjust downward.
- **Filing year 2010 is unexplained**, showing the under-capture signature inside the reliable
  window.
- **Company identifiers are anonymised at source**, which blocks any join to market data, filings
  text or an external default register at the name level.

### A note on seeds

Fold-level Gini on ~30 defaults carries a seed standard deviation of 0.004–0.008, the same size as
many effects being measured here. Two conclusions in this project reversed when re-run across five
seeds. `evaluate.py` averages over five seeds by default and every number above is a mean with its
spread. Single-seed comparisons are not evidence.

---

## Reproduce

```bash
pip install -r requirements.txt
# download american_bankruptcy.csv from the Kaggle link above into data/kaggle/
make all
```

| | |
|---|---|
| `make data` | load and check the Kaggle panel |
| `make mirror` | establish the GitHub-vs-Kaggle difference |
| `make labels` | point-in-time labels with censoring |
| `make leakage` | the four-arm evaluation-design experiment |
| `make evaluate` | walk-forward protocol and the label-capture test |
| `make ladder` | the four-rung model comparison |
| `make calibrate` | calibrated PDs, master scale, migration matrix |
| `make tail` | calibrator comparison + out-of-time overlay |
| `make watchlist` | scored FY2018 book with reason codes |
| `make reasons` | SHAP reason codes and the conceptual-soundness check |
| `make register` | the label-capture finding against US Courts statistics |
| `make figures` | rebuild the README charts from the output CSVs |
| `make dashboard` | build docs/index.html |
| `make test` | pytest |

`make all` runs the pipeline end to end. On two cores it takes about 100 minutes; `make evaluate`
alone is 45 of them, because it fits 280 models — two feature sets × three training windows ×
five seeds × seven annual folds.

## Layout

```
src/
  data.py            loads the Kaggle copy; refuses to silently substitute the GitHub one
  mirror_check.py    establishes how the two copies differ
  labels.py          point-in-time labels, censoring, exit taxonomy, folds
  features.py        credit ratio library: levels, trajectories, industry-relative
  leakage.py         the four-arm evaluation-design experiment
  evaluate.py        walk-forward protocol and the label-capture test
  ladder.py          the four-rung model comparison
  calibrate.py       isotonic calibration, 21-notch master scale, migration matrix
  tail_calibration.py  four calibrators + the out-of-time conservatism overlay
  reasons.py         SHAP reason codes and the conceptual-soundness check
  watchlist.py       the scored FY2018 book the dashboard renders
  external_register.py  the label-capture finding against US Courts statistics
notebooks/
  make_notebook.py   generates the public notebook, so it cannot drift from the repo
  label-leakage-in-the-american-bankruptcy-dataset.ipynb
docs/
  make_figures.py    rebuilds the README charts from the output CSVs
  build_dashboard.py injects the scored book into the page template
  index.html         the live watchlist, served by GitHub Pages
tests/
  test_pipeline.py   24 invariants the panel, the labels and the models must satisfy
  test_readme_numbers.py  asserts every figure quoted above against the table that made it
```

### The README is under test

`tests/test_readme_numbers.py` asserts each headline number in this file against the result
table it came from, and the result tables are tracked, so the 22 checks run from a clean clone
without the dataset present. It exists because the failure mode it guards against happened
here: the repo was rebuilt on a different copy of the data, every number moved, and the prose,
the charts and a published notebook went on quoting the old ones. A figure in this README that
no table supports is now a failing test rather than something a reader has to catch.

## Citation

Pellegrino, M., Lombardo, G., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022).
*Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.*
Future Internet, 14(8), 244.

Nothing here criticises that paper's method, which uses obligor-disjoint splits and an out-of-time
test set. The findings concern the label construction in the distributed data, the notebook
ecosystem built on it, and the difference between the two copies in circulation.

## Licence

MIT.
