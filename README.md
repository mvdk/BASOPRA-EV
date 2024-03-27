# BASOPRA-EV
BASOPRA - BAttery Schedule OPtimizer for Residential Applications. Daily battery schedule, electric vehicle charging, and heat pump demand optimizer (i.e. 24 h optimization framework), assuming perfect day-ahead forecast of the electricity demand load and solar PV generatiom. 

The software was written in python 3.7, please use this version.

The python environment needed to run the program is in the file "basopra env.txt"

Once the setup has been succesfully run, you can run the Main script, for this, change the repertory to the subdirectory BASOPRA and run python3 Main.py:

cd BASOPRA
python3 Main.py

Please have in mind that you will need CPLEX (or gurobi, glpk...) to run the optimization. If you have problems with CPLEX or other optimization software but you are sure you have it, go to Core_LP.py and be sure that the executable path is the correct one for your system, it is in the line opt = SolverFactory('PATH_TO_YOUR_OPTIMIZATION_SOFTWARE')

---------------------------HOW TO USE---------------------

There are two folders: Core and Input.

Core contains the code in five scripts: main_paper_dec.py (main script from which the code can be run), Core_LP.py (setting op the optimization), LP.py (the optimization problem formulation), paper_classes.py (battery characteristics), and post_proc.py (post processing of model results)

Input contains the data input files, including generation and demand profiles for the PV installation, the household load, the electric vehicle, and heat pump and other technical specifications. Several files need to be unzipped before the program can run.

In case of doubts, bugs or problems please contact us by e-mail: marten.vanderkam@unibas.ch or using github.com

A paper explaining the model in detail is in preparation
