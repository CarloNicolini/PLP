.PHONY: help docs-sync docs docs-strict docs-serve docs-clean test

UV ?= uv

DOCS_SRC := docs
DOCS_OUT := docs/_build/html

DOCS_HOST ?= 127.0.0.1
DOCS_PORT ?= 8000

help:
	@echo "Common targets:"
	@echo "  make docs-sync    # uv sync --group docs"
	@echo "  make docs         # sphinx-build (fast iteration)"
	@echo "  make docs-strict  # sphinx-build -W (CI-style)"
	@echo "  make docs-serve   # live-reload dev server (sphinx-autobuild)"
	@echo "  make docs-clean   # remove $(DOCS_OUT)"
	@echo "  make test         # pytest tests/"

docs-sync:
	$(UV) sync --group docs

docs: docs-sync
	$(UV) run --group docs sphinx-build -b html $(DOCS_SRC) $(DOCS_OUT)

docs-strict: docs-sync
	$(UV) run --group docs sphinx-build -W -b html $(DOCS_SRC) $(DOCS_OUT)

docs-serve: docs-sync
	$(UV) run --group docs sphinx-autobuild $(DOCS_SRC) $(DOCS_OUT) \
		--host $(DOCS_HOST) --port $(DOCS_PORT) --open-browser

docs-clean:
	rm -rf $(DOCS_OUT)

test:
	$(UV) run pytest tests/
