This project aims to develop different optimal paths for citizens with mobility issues in the city of Amsterdam by using interactive strategies to obtain utilities from the user. This project is developed on Multi-Objective Optimization using non-parametric approaches such as Gaussian Processes with Expected Improvement. I aim to solve the problem using a parametric approach such as Bayesian Logistic Regression with Thompson Sampling. 

What to run:
1. To run the experiments for Expected Improvement, run the ```run_GP-EI.py``` file in the ```experiments``` folder and change the ```num_iter``` to the desired number of iterations the notebook should be run.
2. To run the experiments for Thompson Sampling, run the ```run_GP-TS.py``` file in the ```experiments``` folder change the ```num_iter``` to the desired number of iterations the notebook should be run.
3. To run Gaussian Processes with Expected Improvement or Thompson Sampling using a different dataset with more or less objectives and datapoints, modify the ```num_objectives``` and path to ```synthetic_pcs``` in ```GP_EI.ipynb``` and ```GP_TS.ipynb``` files respectively. Please also change the path to output the csv file for ```regret``` and result of the EI or TS method.
4. To generate synthetic Pareto Coverage Sets with different objectives and datapoints, run the ```pointset.py``` file in the ```synthetic_pcs_sets``` folder and modify the ```numobjectives``` and ```numpoints``` variables.

