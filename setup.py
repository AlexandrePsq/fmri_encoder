import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fmri_encoder",
    version="0.0.1",
    author="Alexandre Pasquiou",
    author_email="alexandre.pasquiou@inria.fr",
    description="Python implementation of fMRI linear encoding models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexandrePsq/fmri_encoder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["nilearn", "sklearn", "nibabel", "torch", "scipy", "PyYAML"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
