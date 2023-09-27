from setuptools import setup

setup(name='hundred_hammers',
      version='1.0.5',
      description='Batch execution of ML models to a dataset',
      url='http://github.com/vgarciasc/hundred_hammers',
      author='Vinicius Garcia',
      author_email='vinizinho@vinizinho.net',
      license='MIT',
      packages=['hundred_hammers'],
      install_requires=[
          "seaborn", "matplotlib", "adjustText",
          "scikit-learn", "scikit-elm", "xgboost>=1.7.6",
          "pandas", "tqdm", "dask", "dask[distributed]",
          "schema", "pytest"],
      zip_safe=False)
