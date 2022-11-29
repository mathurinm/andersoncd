WARNING
=======
This code is no longer maintained. The codebase has been moved to https://github.com/scikit-learn-contrib/skglm.
This repository only serves to reproduce the results of the AISTATS 2021 paper "Anderson acceleration of coordinate descent" by Quentin Bertrand and Mathurin Massias.




Anderson extrapolation for Coordinate Descent
=============================================

|image0| |image1|


This package implements various GLM models missing from scikit-learn (Lasso, Weighted Lasso, MCP) with a lightning fast solver based on Anderson acceleration of coordinate descent and working set techniques.



Install
=======

To be able to run the code you first need to run, in this folder (code folder, where the setup.py is):
::

    pip install -e .



Reproduce all experiments
=========================

The experiments are on a legacy branch, at https://github.com/mathurinm/andersoncd/tree/backup_expes_paper


- Figure 1: ``expes_ols/plot_intro_ols.py`` (less than 2 min)
- Figure 2: ``expes_eigvals/plot_rayleigh.py`` (around 10 minutes, `main_raileygh.py` must be run first)
- Figure 3: ``expes_cd_sym/plot_cd_sym.py``, choosing `pb = "lasso"` (yes, even if the problem is OLS)  (1 minute)
- Figure 4: ``expes_cd_sym/plot_cd_sym.py``, choosing `pb = "logreg"`  (1 minute)
- Figure 5: ``expes_ols/plot_influ_reg.py`` (less than 2 min)
- Figure 6: ``expes_ols/plot_influ_K.py`` (less than 2 min)


Figure 7, 8, 9 and 10, 11, 12 in appendix (convergence on various datasets and regularizers):
These take quite a long time to run given the extensive validation, they should be run
with multithreading.
::

    cd expes/expe_conv_logreg
    run main.py
    run plot.py

and the same for ``expes/expe_conv_enet`` and ``expes/expe_conv_lasso``


.. |image0| image:: https://github.com/mathurinm/andersoncd/workflows/build/badge.svg
   :target: https://github.com/mathurinm/andersoncd/actions?query=workflow%3Abuild
.. |image1| image:: https://codecov.io/gh/mathurinm/andersoncd/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mathurinm/andersoncd
