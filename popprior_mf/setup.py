from setuptools import setup, find_packages

setup(
    name='popprior_mf',
    version='1.0.0',
    description='Matrix factorization with population priors.',
    author='De-identified author',
    author_email='de-identified@xxx.xxx',
    packages=find_packages(),
    install_requires=[
        'setuptools==59.5.0',
        'grpcio',
        'networkx',
        'torch',
        'torchvision',
        'torchaudio',
        'tqdm',
        'pandas',
        'pyyaml',
        'scikit-learn==0.24.2',
        'matplotlib',
        'seaborn',
        'umap-learn',
        'tensorboard',
        'anndata',
    ],
)
