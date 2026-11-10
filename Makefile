PYTHON ?= .venv/bin/python

.PHONY: setup doctor demo test replay
setup:
	bash scripts/setup_m1.sh
doctor:
	$(PYTHON) -m self_instruct doctor
demo:
	bash scripts/run_m1_demo.sh
test:
	$(PYTHON) -m pytest -q
replay:
	$(PYTHON) -m self_instruct run --config configs/m1_demo.json --output runs/replay --no-train --replay-from experiments/m1-final/requests.jsonl
