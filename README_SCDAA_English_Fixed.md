# SCDAA Coursework 2025 – Relaxed Entropy Regularized LQR: Actor-Critic Implementation

This repository contains a complete implementation of actor-critic methods for solving a continuous-time relaxed entropy regularized Linear Quadratic Regulator (LQR) problem. The implementation follows the specification provided in the SCDAA Coursework 2024–25 at the University of Edinburgh.

## Project Overview
The coursework is organized into five stages corresponding to five exercises. Each stage builds upon the previous one and serves as a milestone toward the final goal of training an actor-critic reinforcement learning agent:

| Exercise | Description |
|----------|-------------|
| 1.1 & 1.2 | Solve strict LQR using Riccati ODE and validate via Monte Carlo simulation |
| 2.1 | Solve the soft (relaxed) LQR problem with entropy regularization |
| 3.1 | Implement a critic-only algorithm to learn the value function |
| 4.1 | Implement an actor-only algorithm to learn the optimal policy |
| 5.1 | Combine actor and critic into a full actor-critic algorithm |

## File Structure
- Ex1_1.py – LQRSolver class and Riccati solution
- Ex1_2.py – Monte Carlo convergence check for strict LQR
- Ex2_1.py – Soft LQR solver with entropy regularization
- Ex2_2.py – Trajectory comparison between strict and soft LQR
- Ex3.py – Critic-only implementation
- Ex4.py – Actor-only implementation
- Ex5.py – Full actor-critic implementation

## Dependencies
Use only the libraries permitted by the coursework:
- numpy
- scipy
- matplotlib
- torch

## How to Run
Each file corresponds to one stage and can be run independently:
```bash
python Ex1_1.py
python Ex1_2.py
python Ex2_1.py
python Ex2_2.py
python Ex3.py
python Ex4.py
python Ex5.py
```

## Group Members
| Name | Student ID | Contribution |
|------|------------|--------------|
| Alice Example | s1234567 | 1/3 |
| Bob Example | s2345678 | 1/3 |
| Charlie Example | s3456789 | 1/3 |

## Notes
- All matrix parameters (H, M, C, D, R, sigma) follow the values provided in the coursework.
- Riccati equations solved using scipy’s solve_ivp.
- Actor and critic are trained offline using PyTorch.