from setuptools import find_packages, setup

package_name = "nav_explain_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amar Halilovic",
    maintainer_email="amar.halilovic@uni-ulm.de",
    description="Core algorithms for explainable robot navigation (ROS-free).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
