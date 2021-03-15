all: install get_data

install:
	pip install -r requirements.txt
	pip install --force-reinstall --no-binary :all: horovod==0.21.0 
	pip uninstall pillow -y
	CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd

get_data:
	cd datasets && ./get_images.sh

test:
	pytest -n 10 --forked
