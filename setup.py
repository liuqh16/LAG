"""Setups up the LAG module."""

from setuptools import setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Get the version ofr Light Aircraft Game."""
    path = "lag/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


version = get_version()
header_count, long_description = get_description()


setup(
    name="lag",
    version=version,
    author="Qihan Liu & Yuhua Jiang",
    author_email="liuqh.thu@gmail.com",
    description="An environment based on JSBSIM aimed at 1v1 and 2v2 air combat game.",
    url="https://github.com/liuqh16/LAG",
    license="GPL",
    license_files=("LICENSE"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=["Environment", "Multi-Agent", "RL", "Gym", "Aircraft", "Combat"],
    python_requires=">=3.6",
    packages=["lag", "lag.envs.jsbsim"],
    package_data={
        "lag.envs.jsbsim": [
            "model/*.pt",
        ]
    },
    include_package_data=True,
    install_requires=[
        "jsbsim==1.2.1",
        "gymnasium==0.28.1"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)