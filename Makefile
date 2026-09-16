PY := PYTHONPATH=src python3
SRC := https://github.com/sowide/bankruptcy_dataset.git

.PHONY: all data audit repair labels evaluate ladder calibrate tail watchlist dashboard register leakage verify notebook test clean

all: repair labels evaluate ladder calibrate tail watchlist

data: data/raw_sowide/american_bankruptcy_dataset.csv

data/raw_sowide/american_bankruptcy_dataset.csv:
	git clone --depth 1 $(SRC) data/raw_sowide
	cd data/raw_sowide && unzip -o -q dataset_paper.zip -d ../paper

audit: data
	$(PY) src/audit.py

repair: data
	$(PY) src/repair.py

labels: outputs/clean_panel.parquet
	$(PY) src/labels.py

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

dashboard: outputs/dashboard_data.json
	$(PY) docs/build_dashboard.py

notebook:
	$(PY) notebooks/make_notebook.py

register: outputs/labelled_panel.parquet
	$(PY) src/external_register.py

leakage: data
	$(PY) src/leakage.py

verify: outputs/clean_panel.parquet
	$(PY) src/verify_repair.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -f outputs/*.parquet outputs/*.csv
