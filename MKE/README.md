Solving partial differential equations using fenics
===

Implementation of the solution of two types of partial differential equations

## Boundary problem

Problem:

<img src="boundary.PNG" width="300" height="100">

Results:

<img src="1.PNG" width="300">
<img src="2.PNG" width="300">
<img src="3.PNG" width="300">

## Heat conduction problem:

<img src="heat.PNG" width="400" height="200">

Results:

Error plots in test.ipynb

<img src="1.gif" width="200" align="left">
<img src="2.gif" width="200" align="center">
<img src="3.gif" width="200" align="right">


Implemetation:
------------
* test.ipynb - main notebook with all results
* PlotSolutions.py - class that consists all function for plot results
* BoundaryProblem.py - implementation of solving boundary problem 
* HeatProblem.py - implementation of solving heat conduction problem 