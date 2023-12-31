import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlgebra",
    version="0.2.0",
    author="Ahmet Erdem",
    description="A machine learning tool for Python, in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.9',
    py_modules=["mlgebra"],
    install_requires=["vectorgebra>=2.7.1"]
)