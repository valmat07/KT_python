Solving partial differential equations using fenics
===

Implementation of the solution of two types of partial differential equations

## Boundary problem

Problem:

<img src="pics/boundary.PNG" width="300" height="100">

Results:

<img src="pics/1.PNG" width="300">
<img src="pics/2.PNG" width="300">
<img src="pics/3.PNG" width="300">

## Heat conduction problem:

<img src="pics/heat.PNG" width="400" height="200">

Results:

Error plots in test.ipynb

<img src="pics/1.gif" width="300">
<img src="pics/2.gif" width="300">
<img src="pics/3.gif" width="300">



Implemetation:
------------
* test.ipynb - main notebook with all results
* PlotSolutions.py - class that consists all function for plot results
* BoundaryProblem.py - implementation of solving boundary problem 
* HeatProblem.py - implementation of solving heat conduction problem 