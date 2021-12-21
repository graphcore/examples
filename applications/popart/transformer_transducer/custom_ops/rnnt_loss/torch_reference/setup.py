from setuptools import setup, Extension
import sys
import os
from torch.utils import cpp_extension


if __name__ == '__main__':
    if sys.platform == "darwin":
        args = ["-DAPPLE"]
    else:
        args = ["-fopenmp"]
    if os.environ.get("DEBUG_TRANSDUCER") == "1":
        args.append("-DDEBUG")
        args.append("-O0")

    ext = Extension(
            name='transducer_cpp',
            sources=['transducer.cpp'],
            include_dirs=cpp_extension.include_paths(),
            extra_compile_args=args,
            language='c++')

    setup(name='transducer_cpp',
            ext_modules=[ext],
            cmdclass={'build_ext': cpp_extension.BuildExtension})

