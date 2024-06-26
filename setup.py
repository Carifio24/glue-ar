import os
import sys

sys.path.append(".")
from setupbase import (
    create_cmdclass,
    install_npm,
    combine_commands,
)

from setuptools import setup

def data_files(root_directory):
    paths = []
    for (path, _directories, filenames) in os.walk(root_directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths

name = "glue_ar"

js_content_command = combine_commands(
    install_npm("js", build_cmd="glue-ar-export")
)

# Custom "command class" that (1) makes sure to create the JS content, (2)
# includes that content as extra "package data" in the Python package, and (3)
# can install special metadata files in the Python environment root.
data_files_spec = [
    (".", name, "*.png"),
    (".", name, "*.ui"),
    ("js", "js", "**/*.*")
]

cmdclass = create_cmdclass(
    "js-content", data_files_spec=data_files_spec
)
cmdclass["js-content"] = js_content_command

# TODO: Add the rest of the package data
setup_args = dict(
    name=name,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    zip_safe=False,
    packages=[name],
    include_package_data=True,
    install_requires=[
        "gltflib",
        "glue-core",
        "glue-qt",
        "glue-vispy-viewers",
        "ngrok",
        "pillow",
        "pyvista",
        "segno"
    ],
    extras_require={
        "test": [
            "flake8"
        ]
    },
    entry_points={
        "glue.plugins": "glue_ar = glue_ar:setup"
    }
)

if __name__ == "__main__":
    setup(**setup_args)
