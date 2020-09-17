CS 170 Fall 2019 Project

Ismail Javed

Meera Mehta

Vidhan Jain

Drive The TA’s Home: Approximation Algorithm for an NP-Hard Problem

The solver uses an integer linear programming approach and some additional heuristics as explained in the final report to compute the output. The ILP solver is obtained from gurobi.com and can be specifically found here - http://examples.gurobi.com/traveling-salesman-problem/
The download and license activation page is this - https://www.gurobi.com/downloads/

A few additional steps before the code can be run:

1. Install Gurobi from the main website and obtain a license (free academic license)
1. Install the Python Interface for Gurobi. This is most easily achieved by installing Anaconda (a version compatible with Python 3.7), and using the Anaconda terminal to install Gurobi (instructions can be found online).
1. Make sure in the Python / Conda environment being used to run the code the appropriate packages used for the code are installed
1. Run the solver on an input from the command line 
  1. python solver.py [path-to-input-file] [output-directory]
  1. python solver.py --all [path-to-input-directory] [output-directory]

