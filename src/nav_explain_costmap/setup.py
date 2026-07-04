import os
from glob import glob
from setuptools import find_packages, setup

package_name = "nav_explain_costmap"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amar Halilovic",
    maintainer_email="amar.halilovic@uni-ulm.de",
    description="Costmap explanation methods (LIME/SHAP/Anchors/counterfactual).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "nec_explain = nav_explain_costmap.apps.explain_scenario:main",
        "nec_figures = nav_explain_costmap.apps.reproduce_figures:main",
        "nav_explain_costmap_node = nav_explain_costmap.ros.node:main",
    ]},
)
