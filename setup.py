from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cotkg-network-intrusion-detection",
    version="0.1.1",
    author="Chen Xingqiang",
    author_email="chen.xingqiang@iechor.com",
    description="A network intrusion detection system using Chain of Thought, knowledge graphs and GraphSAGE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenxingqiang/cotkg-network-intrusion-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "networkx",
        "py2neo",
        "matplotlib",
        "seaborn",
        "tsfresh",
        "shap",
        "transformers",
        "openai",
        "tqdm",
    ],
)
