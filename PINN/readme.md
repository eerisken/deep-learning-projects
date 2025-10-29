
**Physics-Informed Neural Network for 1D Heat Equation**

This Python code uses the PyTorch library to create and train a Physics-Informed Neural Network (PINN). The goal is to find an approximate solution to the 1D heat equation (ut​=αuxx​) over a specified time and space domain.

It works by training a neural network not just on data points, but also by making sure its output satisfies the heat equation itself, as well as the given initial temperature distribution (t=0) and boundary conditions (at x=0 and x=L). Automatic differentiation (autograd) is used to calculate the derivatives (u_t​, u_xx​) needed to check if the network's output respects the physics of the heat equation.
