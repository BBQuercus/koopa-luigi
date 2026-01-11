.PHONY: test test-setup test-run test-verify clean help update-koopa update-koopa-legacy update-koopa-modern

# Project root directory
ROOT := $(shell pwd)

# Test directories
TEST_DIR := $(ROOT)/tests
TEST_OUTPUT := $(TEST_DIR)/output
TEST_CONFIG := $(TEST_OUTPUT)/koopa_test.cfg
FIXTURES_DIR := $(TEST_DIR)/fixtures

# Default target
help:
	@echo "Koopa-Luigi Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make test              Run the full test suite"
	@echo "  make test-setup        Set up test environment only"
	@echo "  make clean             Remove test outputs"
	@echo "  make update-koopa      Update koopa in both environments"
	@echo "  make update-koopa-legacy   Update koopa in venv_legacy only"
	@echo "  make update-koopa-modern   Update koopa in venv_modern only"
	@echo "  make help              Show this help message"

# Full test pipeline (preserves cache for Luigi resumability)
test: test-setup test-run test-verify
	@echo ""
	@echo "=========================================="
	@echo "  All tests passed!"
	@echo "=========================================="

# Set up the test environment
test-setup:
	@echo "Setting up test environment..."
	@mkdir -p $(TEST_OUTPUT)/preprocessed
	@# Copy preprocessed fixture (so Luigi skips preprocessing)
	@cp $(FIXTURES_DIR)/preprocessed/A_H_1_2.tif $(TEST_OUTPUT)/preprocessed/
	@# Generate config with absolute paths
	@sed 's|__ROOT__|$(ROOT)|g' $(TEST_DIR)/koopa_test.cfg > $(TEST_CONFIG)
	@echo "Test environment ready"
	@echo "  Config: $(TEST_CONFIG)"
	@echo "  Output: $(TEST_OUTPUT)"

# Python interpreter (prefer venv, then python3, then python)
PYTHON := $(shell if [ -f "$(ROOT)/.venv/bin/python" ]; then echo "$(ROOT)/.venv/bin/python"; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)

# Run the pipeline
test-run:
	@echo ""
	@echo "Running koopa-luigi pipeline..."
	@echo ""
	$(PYTHON) -m src --config $(TEST_CONFIG) --workers 2 --skip-incompatible
	@echo ""

# Verify outputs exist
test-verify:
	@echo "Verifying test outputs..."
	@test -f $(TEST_OUTPUT)/summary.csv || (echo "FAIL: summary.csv not found" && exit 1)
	@test -f $(TEST_OUTPUT)/summary_cells.csv || (echo "FAIL: summary_cells.csv not found" && exit 1)
	@test -f $(TEST_OUTPUT)/koopa.log || (echo "FAIL: koopa.log not found" && exit 1)
	@test -d $(TEST_OUTPUT)/segmentation_nuclei || (echo "FAIL: segmentation_nuclei/ not found" && exit 1)
	@test -d $(TEST_OUTPUT)/segmentation_cyto || (echo "FAIL: segmentation_cyto/ not found" && exit 1)
	@test -d $(TEST_OUTPUT)/detection_raw_c0 || (echo "FAIL: detection_raw_c0/ not found" && exit 1)
	@# Check summary.csv has content (more than just header)
	@test $$(wc -l < $(TEST_OUTPUT)/summary.csv) -gt 1 || (echo "FAIL: summary.csv is empty" && exit 1)
	@echo "All outputs verified successfully"

# Clean up test outputs
clean:
	@echo "Cleaning test outputs..."
	@rm -rf $(TEST_OUTPUT)
	@echo "Clean complete"

# Update koopa in both environments (--no-deps to preserve pinned numpy)
update-koopa: update-koopa-legacy update-koopa-modern
	@echo "Koopa updated in both environments"

update-koopa-legacy:
	@echo "Updating koopa in venv_legacy..."
	uv pip install --upgrade koopa --no-deps --python $(ROOT)/venv_legacy/bin/python

update-koopa-modern:
	@echo "Updating koopa in venv_modern..."
	uv pip install --upgrade koopa --no-deps --python $(ROOT)/venv_modern/bin/python
