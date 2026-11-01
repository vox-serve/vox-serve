"""setuptools entry point; the package is configured in pyproject.toml.

It adds one thing: the Rust extension vox_serve._rt is an optional build, and a failed build must not leave the
copy of an earlier build in place (setuptools' build/ directory, or the source tree for editable installs).
`vox-serve` would load that stale extension with the current Python code instead of falling back.
"""

import glob
import os

from setuptools import setup
from setuptools_rust.build import build_rust


class BuildRust(build_rust):
    def run(self):
        # Before anything else: without a Rust compiler, build_rust returns before building any extension.
        build_lib = self.get_finalized_command("build_ext").build_lib
        for ext in self.extensions:
            for root in [build_lib, "."] if self.inplace else [build_lib]:
                for path in glob.glob(os.path.join(root, *ext.name.split(".")) + ".*so"):
                    os.remove(path)
        super().run()


# The extension uses the stable ABI from Python 3.12 (pyo3 feature abi3-py312): one wheel per platform.
setup(cmdclass={"build_rust": BuildRust}, options={"bdist_wheel": {"py_limited_api": "cp312"}})
