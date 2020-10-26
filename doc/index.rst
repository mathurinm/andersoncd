.. andersoncd documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

andersoncd
==========

This is a library to run Anderson extrapolated coordinate descent.

Installation
------------
First clone the repository available at https://github.com/mathurinm/andersoncd::

    $ git clone https://github.com/andersoncd.git
    $ cd andersoncd/


We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.

From a working environment, you can install the package with::

    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import andersoncd'

and it should not give any error message.

From a Python shell you can just do::

    >>> import andersoncd

If you don't want to use Anaconda, you should still be able to install using `pip`.


API
---

.. toctree::
    :maxdepth: 1

    api.rst
