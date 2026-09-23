PY     := PYTHONPATH=src python3
KAGGLE := data/kaggle/american_bankruptcy.csv
MIRROR := https://github.com/sowide/bankruptcy_dataset.git

.PHONY: all data mirror labels leakage evaluate ladder calibrate tail watchlist reasons \
        dashboard figures register notebook test clean

all: data labels leakage evaluate ladder calibrate tail watchlist reasons register figures dashboard

# ---------------------------------------------------------------- data
# The Kaggle file is NOT downloadable without credentials and is not redistributed
# here. Download american_bankruptcy.csv from
#   kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset
# and drop it in data/kaggle/. src/data.py refuses to substitute the GitHub copy.
$(KAGGLE):
	@echo "Missing $(KAGGLE) - see the note in the Makefile and src/data.py."; exit 1

data: $(KAGGLE)
	$(PY) src/data.py

# the GitHub copy, used only for the SIC sector codes and for mirror_check
data/raw_sowide/american_bankruptcy_dataset.csv:
	git clone --depth 1 $(MIRROR) data/raw_sowide

mirror: $(KAGGLE) data/raw_sowide/american_bankruptcy_dataset.csv
	$(PY) src/mirror_check.py

# ---------------------------------------------------------------- pipeline
labels: outputs/clean_panel.parquet
	$(PY) src/labels.py

leakage: outputs/clean_panel.parquet
	$(PY) src/leakage.py

evaluate: outputs/labelled_panel.parquet
	$(PY) src/evaluate.py

ladder: outputs/labelled_panel.parquet
	$(PY) src/ladder.py

calibrate: outputs/labelled_panel.parquet
	$(PY) src/calibrate.py

tail: outputs/labelled_panel.parquet
	$(PY) src/tail_calibration.py

watchlist: outputs/labelled_panel.parquet
	$(PY) src/watchlist.py

reasons: outputs/labelled_panel.parquet
	$(PY) src/reasons.py

register: outputs/labelled_panel.parquet
	$(PY) src/external_register.py

# ---------------------------------------------------------------- outputs
figures:
	$(PY) docs/make_figures.py

dashboard: outputs/dashboard_data.json
	$(PY) docs/build_dashboard.py

notebook:
	$(PY) notebooks/make_notebook.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -f outputs/*.parquet outputs/*.csv outputs/*.log
