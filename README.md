# dnosearch

Source code for Deep Neural Operator (DeepONet) Bayesian experimental design (i.e. active learning) and optimization with likelihood-weighted acquisition functions. 

## Installation Guide

Execute `pip install .` from the master directory (directory where setup.py is located).

This package requires the use of DeepONet from the DeepXDE library. While installing these requirements, it may be useful to comment the deepxde package and install separately. If using a verion of DeepXDE above v0.11.2, then DeepONet calls must be changed approprately, as stated by DeepXDEs instructions done separately. 

## Demo

Three folders in the examples directory: SIR, NLS, LAMP provide the basis for computing any of the results from the paper. The SIR folder provides the simplest, yet complete demonstration of the algorithm. We recommend learning from the SIR implementation for applying to novel problems, as the other problems are computationally much more demanding.

SIR.py is setup to run in an IDE (e.g. Spyder) and dynamically plots various results. This is recommended for best understanding of the code and process. SIR_bash.py is set up to be called by the shell script, SIR_shell.sh, and will plot to the SIR directory. The advantage of the shell script is that Tensorflow slows in time, while the shell script reopens Python at each iteration ensuring that Tensorflow runs quickly. 

The MMT(NLS) code requires matlab be called to solve the MMT equations. For smoother passing of the code, the shell scripts perform this, by individually calling matlab at each iteration. This also allows Tensforflow to restart. To recreate the various plots in the study, the shell script must be changed to reflect the parameters of each study. Considering the number of independent experiments and conditions, calculating all will take substantial computational time.

For MMT, several shell scripts are created and labelled for the various figures. Running these scripts will provide data in the results folder for which the plotting file, and associated blocks of the file will create a light version of the figure. For each of these, there is currently data provided for one experiment (1 seed) per case in the data folder of the paper link.


The LAMP code (located in the lamp folder) is divided into the DNO and GP components. DNO is performed by running the bash script LAMP_10D_run.sh. This script can take substantial time to run. The GP code is run by entering the Matlab_GP_Implementation subfolder and running the code: RUN_ME.m

## Data

For some of the code, certain data sets or folders are needed and are rather large and provided at the following links.

The truth data file for the SIR model is located at: https://www.dropbox.com/s/defzt6usnn7m3ij/truth_data.mat?dl=0
IC folder for the MMT solution is located at: https://www.dropbox.com/sh/izrs1n261heivr7/AAD4OBDm-QEnYGVbR4u25o_8a?dl=0
The folder for the LAMP problem is located at: https://www.dropbox.com/sh/93u5ypxzhnxxql8/AADWY-CLBF-aK1hpEUjuR01Ba?dl=0

## References
* [Discovering and forecasting extreme events via active learning in
neural operators](https://arxiv.org/pdf/2204.02488.pdf)
* [Bayesian Optimization with Output-Weighted Optimal Sampling](https://arxiv.org/abs/2004.10599)
* [Informative Path Planning for Anomaly Detection in Environment Exploration and Monitoring](https://arxiv.org/abs/2005.10040)
* [Output-Weighted Optimal Sampling for Bayesian Experimental Design and Uncertainty Quantification](https://arxiv.org/abs/2006.12394)
