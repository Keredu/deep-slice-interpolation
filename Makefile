.PHONY: train train-one train-queue reset-errors lint clean test test-fast report

train: clean
	@status=0; \
	uv run train.py --run-all || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

train-one: clean
	@status=0; \
	uv run train.py || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

train-queue: clean
	@status=0; \
	uv run train.py --show-queue || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

reset-errors: clean
	@status=0; \
	uv run register_experiments.py --reset-errors || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

lint: clean
	@status=0; \
	(uv run ruff check . --fix && uv run ruff format .) || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	
test: clean
	@status=0; \
	uv run pytest || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

test-fast: clean
	@status=0; \
	uv run pytest --no-cov || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

report: clean
	@status=0; \
	uv run report.py || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

loss: clean
	@status=0; \
	uv run scripts/plot_loss_curves.py || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status

test-patients-viz: clean
	@status=0; \
	uv run scripts/generate_missing_viz.py || status=$$?; \
	$(MAKE) --no-print-directory clean; \
	exit $$status
