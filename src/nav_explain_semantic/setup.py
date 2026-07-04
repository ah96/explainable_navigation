import os
from glob import glob
from setuptools import find_packages, setup

package_name = "nav_explain_semantic"


def data_tree(dest, src):
    out = []
    for path in glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(path):
            out.append((os.path.join("share", package_name, dest,
                        os.path.dirname(os.path.relpath(path, src))), [path]))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ] + data_tree("data", "data"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amar Halilovic",
    maintainer_email="amar.halilovic@uni-ulm.de",
    description="Visuo-textual semantic navigation explanations.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "nes_explain = nav_explain_semantic.apps.explain_scenario:main",
        "nav_explain_semantic_node = nav_explain_semantic.ros.node:main",
    ]},
)
