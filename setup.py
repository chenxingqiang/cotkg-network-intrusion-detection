from setuptools import setup, find_packages

setup(
    name='cotkg-network-intrusion-detection',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A network intrusion detection system using Chain of Thought, knowledge graphs and GraphSAGE',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'torch-geometric',
        'networkx',
        'py2neo',
        'tqdm',
        'matplotlib',
        'seaborn',
        'tsfresh',
        'shap'
    ],
    url='https://github.com/chenxingqiang/cotkg-network-intrusion-detection',
    author='Chen Xingqiang',
    author_email='chen.xingqiang@iechor.com'
)
