.PHONY: clean compile

compile:
	python3 -m examples_utils load_lib_build .

clean:
	find . -name '*.so' -delete
	find . -name '*.so.lock' -delete
	find . -name '.rendered.*.cpp' -delete
