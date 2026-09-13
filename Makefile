PY := PYTHONPATH=src python3
SRC := https://github.com/sowide/bankruptcy_dataset.git

.PHONY: all data audit repair labels evaluate leakage verify test clean

all: repair labels evaluate

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

leakage: data
	$(PY) src/leakage.py

verify: outputs/clean_panel.parquet
	$(PY) src/verify_repair.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -f outputs/*.parquet outputs/*.csv
