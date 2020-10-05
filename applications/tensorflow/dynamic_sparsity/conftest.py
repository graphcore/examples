from pathlib import Path
from subprocess import run, PIPE

public_examples_dir = Path(__file__).parent.parent.parent.parent


def rebuild_dynamic_sparsity_custom_ops():
    """This function builds the dynamic_sparsity
    custom ops for any tests that rely on it.
    """
    build_path = Path(
        public_examples_dir,
        "applications",
        "tensorflow",
        "dynamic_sparsity"
    )
    completed = run(['python-config', '--extension-suffix'], stdout=PIPE)
    extension = completed.stdout.decode().replace('\n', '')
    print("\nCleaning dynamic_sparsity")
    run(['make', 'clean'], cwd=build_path)
    print("\nBuilding dynamic_sparsity")
    run(['make', '-j'], cwd=build_path)
    shared_libs = [f'host_utils{extension}', 'libsparse_matmul.so']
    paths = [Path(build_path, "ipu_sparse_ops", f) for f in shared_libs]
    exist = [path.exists() for path in paths]
    assert all(exist)


def pytest_sessionstart(session):
    rebuild_dynamic_sparsity_custom_ops()
