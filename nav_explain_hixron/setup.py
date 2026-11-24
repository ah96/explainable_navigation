import os
from glob import glob

from setuptools import find_packages, setup

package_name = "nav_explain_hixron"


def data_tree(dest: str, src: str) -> list[tuple[str, list[str]]]:
    """Install the files below ``src`` that the tools read to ``share/<package>/<dest>``.

    These are the tables of the scenarios, the recorded results and the study
    answers. The simulation worlds and the maps stay in the source tree: no tool
    reads them, and they are 20 MB.
    """
    out = []
    for path in glob(os.path.join(src, "**", "*.csv"), recursive=True):
        if os.sep + "map" + os.sep in path:
            continue
        folder = os.path.dirname(os.path.relpath(path, src))
        out.append((os.path.join("share", package_name, dest, folder), [path]))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ] + data_tree("data", "data"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amar Halilovic",
    maintainer_email="amar.halilovic@uni-ulm.de",
    description="HiXRoN explanations: scenarios, recorded results, a scripted run and a live node.",
    license="MIT",
    # colcon reads this entry to run the tests with pytest
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": [
        "neh_replay = nav_explain_hixron.apps.replay:main",
        "neh_results = nav_explain_hixron.apps.results:main",
        "nav_explain_hixron_node = nav_explain_hixron.ros.node:main",
    ]},
)
