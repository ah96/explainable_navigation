from setuptools import find_packages, setup

package_name = "nav_explain_perception"

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
    description="YOLO object detection for semantic navigation explanations.",
    license="MIT",
    # colcon reads this entry to run the tests with pytest
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": [
        "nep_detect = nav_explain_perception.apps.detect_image:main",
        "nav_explain_perception_node = nav_explain_perception.ros.node:main",
    ]},
)
