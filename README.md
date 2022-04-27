# dnosearch
  
Source code for Deep Neural Operator (DeepONet) Bayesian experimental design (i.e. active learning) and optimization with likelihood-weighted acquisition functions.

## Installation Guide

Execute `pip install .` from the master directory (directory where setup.py is located).

This package requires the use of DeepONet from the DeepXDE library. While installing these requirements, it may be useful to comment the deepxde package and install separately. If using a verion of DeepXDE above v0.11.2, then DeepONet calls must be changed approprately, as stated by DeepXDEs instructions.done separately.

## Demo

Two folders in the examples directory: SIR and NLS, provide the basis for computing any of the results from the paper. The SIR folder provides the simplest, yet complete demonstration of the algorithm.

SIR.py is setup to run in an IDE (e.g. Spyder) and dynamically plots various results. SIR_bash.py is set up to be called by the shell script, SIR_shell.sh, and will plot to the SIR directory. The advantage of the shell script is that Tensorflow slows in time, while the shell script reopens Python at each iteration ensuring that Tensorflow runs quickly.


The NLS code requires matlab be called to solve the NLS equations. For smoother passing of the code, the shell script performs this, by individually calling matlab at each iteration. This also allows Tensforflow to restart. To recreate the various plots in the study, the shell script must be changed to reflect the parameters of each study. Considering the number of independent experiments and conditions, calculating all will take substantial computational time.


## References
* [Discovering and forecasting extreme events via active learning in
neural operators](https://arxiv.org/pdf/2204.02488.pdf)
* [Bayesian Optimization with Output-Weighted Optimal Sampling](https://arxiv.org/abs/2004.10599)
* [Informative Path Planning for Anomaly Detection in Environment Exploration and Monitoring](https://arxiv.org/abs/2005.10040)
* [Output-Weighted Optimal Sampling for Bayesian Experimental Design and Uncertainty Quantification](https://arxiv.org/abs/2006.12394)                                                                                                                                      
