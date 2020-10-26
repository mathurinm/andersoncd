from setuptools import setup


setup(name='andersoncd',
      install_requires=['celer>=0.5.1', 'numpy>=1.12', 'numba', 'seaborn>=0.7',
                        'joblib', 'scipy>=0.18.0', 'matplotlib>=2.0.0',
                        'scikit-learn>=0.23', 'pandas', 'ipython'],
      packages=['andersoncd'],
      )
