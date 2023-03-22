# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install lint test clean

install:
	pip3 install -r requirements.txt
	pip3 install -r requirements-dev.txt
	make -C data_utils/feature_generation
	make -C static_ops

lint:
	yapf --recursive --in-place .
	python3 ci/test_copyright.py

test:
	pytest --forked -n 10

clean:
	find . -name '*.so' -delete
	find . -name '*.lockfile' -delete
