extracd
=======

This package implements Anderson extrapolation for coordinate descent.

It also implements inertial proximal gradient descent (ie APPROXwithout parallelism).

In order to be able to reproduce the experiments you can install the package, by first creating a conda environment.


Install
=====

To be able to run the code you first need to run, in this folder (code folder, where the setup.py is):
```pip install -e .```

If we forgot any dependency, we kindly ask you to install it yourself.

You should now be able to run a friendly example:
```ipython -i expes/expe_ols/plot_intro_ols.py```


Reproduce all experiments
=========================

Figure 1: `expes_ols/plot_intro_ols.py` (less than 2 min)
Figure 2: `expes_eigvals/plot_rayleigh.py` (around 10 minutes, `main_raileygh.py` must be run first)
Figure 3: `expes_cd_sym/plot_cd_sym.py`, choosing `pb = "lasso"` (yes, even if the problem is OLS)  (1 minute)
Figure 4: `expes_cd_sym/plot_cd_sym.py`, choosing `pb = "logreg"`  (1 minute)
Figure 5: `expes_ols/plot_influ_reg.py` (less than 2 min)
Figure 6: `expes_ols/plot_influ_K.py` (less than 2 min)


Figure 7, 8, 9 and 10, 11, 12 in appendix (convergence on various datasets and regularizers):
These take quite a long time to run given the extensive validation, they should be run
with multithreading.
```bash
cd expes/expe_conv_logreg
run main.py
run plot.py
```
and the same for `expes/expe_conv_enet` and `expes/expe_conv_lasso`