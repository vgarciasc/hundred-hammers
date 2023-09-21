.. hundred_hammers documentation master file, created by
   sphinx-quickstart on Thu Sep 21 13:15:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hundred_hammers's documentation!
===========================================

| "At least *one* of them is bound to do the trick."

*Hundred Hammers* is a Python package that helps you batch-test ML models in a dataset. It can be used out-of-the-box
to run most popular ML models and metrics, or it can be easily extended to include your own.

- Supports both classification and regression.
- Already comes strapped with most sci-kit learn models.
- Already comes with several plots to visualize the results.
- Easy to integrate with parameter tuning from GridSearch CV.
- Already gives you the average metrics from training, test, validation (train) and validation (test) sets.
- Allows you to define how many seeds to consider, so you can increase the significance of your results.
- Produces a Pandas DataFrame with the results (which can be exported to CSV and analyzed elsewhere).


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   API reference <hundred_hammers>
