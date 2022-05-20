all: install get-data

install-turbojpeg: get-turbojpeg build-turbojpeg

install:
	pip install -r requirements.txt
	pip uninstall pillow -y
	CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd

get-turbojpeg:
	cd datasets && rm -rf libjpeg-turbo && git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
	cd datasets/libjpeg-turbo && git checkout 0565548 && git apply ../turbojpeg.diff

build-turbojpeg:
	cd datasets/libjpeg-turbo && mkdir -p build
	cd datasets/libjpeg-turbo/build && export CMAKE_PREFIX_PATH="$(shell pwd)/datasets/turbojpeg" && cmake -DCMAKE_INSTALL_PREFIX:PATH="$$CMAKE_PREFIX_PATH" -DCFLAGS="-march=native -O3" ..
	cd datasets/libjpeg-turbo/build && make -j 96 && make install

get-data:
	cd datasets && ./get_images.sh

test:
	cd tests && pytest -n 10 --forked
	cd tests_serial && pytest
