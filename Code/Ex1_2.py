import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from Ex1_1 import LQRSolver  # Import the LQRSolver class from Ex1_1.py

torch.manual_seed(42)  # Set the random seed for reproducibility

def simulate_LQR_explicit_batch(lqr_solver, x0, N, n_samples, device="cpu"):
    """
    Run Monte Carlo simulation using explicit Euler scheme.
    """
    T = lqr_solver.T               # Total simulation time
    tau = T / N                    # Time step size

    # Extract model parameters and convert them to torch tensors on the specified device
    H = torch.tensor(lqr_solver.H, dtype=torch.float32, device=device)      # State dynamics matrix
    M = torch.tensor(lqr_solver.M, dtype=torch.float32, device=device)      # Control input matrix
    sigma = torch.tensor(lqr_solver.sigma, dtype=torch.float32, device=device)  # Noise matrix
    C = torch.tensor(lqr_solver.C, dtype=torch.float32, device=device)      # State cost matrix
    D = torch.tensor(lqr_solver.D, dtype=torch.float32, device=device)      # Control cost matrix
    R = torch.tensor(lqr_solver.R, dtype=torch.float32, device=device)      # Terminal cost matrix
    D_inv = torch.linalg.inv(D)     # Inverse of D for computing optimal control

    # Initialize the state for each sample; repeat initial state x0 n_samples times
    X = torch.tensor(x0, dtype=torch.float32, device=device).repeat(n_samples, 1)
    cost = torch.zeros(n_samples, dtype=torch.float32, device=device)  # Initialize running cost for each sample
    t = 0.0  # Initialize simulation time

    for step in range(N):
        # Get the value function matrix S(t) at time t from the solver
        S_t, _ = lqr_solver._get_S_at_time(t)
        S_t = torch.tensor(S_t, dtype=torch.float32, device=device)  # Convert S_t to a torch tensor
        S_batch = S_t.unsqueeze(0).expand(n_samples, -1, -1)  # Expand S_t to match the batch size

        x_vec = X.unsqueeze(2)  # Reshape state vector for matrix multiplication (n_samples x state_dim x 1)
        a = -D_inv @ M.T @ (S_batch @ x_vec)  # Compute optimal control: a = -D⁻¹ Mᵀ S(t) x
        a = a.squeeze(2)  # Remove the last singleton dimension to get shape (n_samples, state_dim)

        # Calculate running cost: cost += tau * (xᵀ C x + aᵀ D a)
        cost += tau * ((X @ C * X).sum(dim=1) + (a @ D * a).sum(dim=1))

        # Simulate the state update using the Euler-Maruyama method
        dW = torch.randn(n_samples, 2, device=device) * math.sqrt(tau)  # Generate random noise scaled by sqrt(tau)
        drift = X @ H.T + a @ M.T  # Compute drift from state dynamics and control
        X = X + tau * drift + dW @ sigma.T  # Update state with drift and diffusion (noise)
        t += tau  # Increment time by one time step

    # Add the terminal cost: cost += xᵀ R x
    cost += (X @ R * X).sum(dim=1)
    return cost.mean().item()  # Return the mean cost over all samples

def simulate_LQR_avg(lqr_solver, x0, N, n_samples, n_repeats=3):
    """
    Repeat simulation to reduce variance and average result.
    """
    costs = []  # List to store simulation cost from each repeat
    for i in range(n_repeats):
        torch.manual_seed(73 + i)  # Set a different seed for each repeat to vary the simulations
        cost = simulate_LQR_explicit_batch(lqr_solver, x0, N, n_samples)  # Run simulation for current seed
        costs.append(cost)  # Append the obtained cost
    return float(np.mean(costs))  # Return the average cost over all repeats

def experiment_time_steps_stable(lqr_solver, x0, n_samples=10000, n_repeats=3):
    """
    Run convergence experiment by varying time steps (N).
    """
    print("\nExperiment A: Varying Time Steps")
    # Compute the analytic cost using the value function from the LQR solver
    analytic_cost = lqr_solver.value_function(
        torch.tensor([0.0]), torch.tensor([x0], dtype=torch.float32)
    ).item()

    N_list = [2 ** k for k in range(1, 12)]  # List of different numbers of time steps (powers of 2)
    errors = []  # List to store error for each time step count

    for N in N_list:
        sim_cost = simulate_LQR_avg(lqr_solver, x0, N, n_samples, n_repeats)  # Simulated cost with current N
        error = abs(sim_cost - analytic_cost)  # Absolute error between simulated and analytic cost
        errors.append(error)
        print(f"N = {N:4d}, Simulated = {sim_cost:.5f}, Analytic = {analytic_cost:.5f}, Error = {error:.5f}")

    # Plot the error versus number of time steps on a log-log scale
    plt.figure()
    plt.loglog(N_list, errors, marker='o', label='Absolute Error')
    plt.xlabel('Number of Time Steps (N)')
    plt.ylabel('Error')
    plt.title('Convergence vs Time Steps')
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.show()

    return N_list, errors

def experiment_samples_batchwise_stable(lqr_solver, x0, N=10000, sample_exponents=range(10, 16)):
    """
    Run convergence experiment by varying number of Monte Carlo samples.
    """
    print("\nExperiment B: Varying Monte Carlo Samples")
    # Compute the analytic cost using the value function from the LQR solver
    analytic_cost = lqr_solver.value_function(
        torch.tensor([0.0]), torch.tensor([x0], dtype=torch.float32)
    ).item()

    sample_list = [2 ** k for k in sample_exponents]  # List of different sample sizes (powers of 2)
    errors = []  # List to store empirical standard error for each sample size
    theory = []  # List to store theoretical standard error (1/sqrt(N) reference)

    for i, n_samples in enumerate(sample_list):
        torch.manual_seed(100 + i)  # Set seed for reproducibility for each sample size
        costs = []  # List to store cost from multiple simulation runs
        for _ in range(10):
            cost = simulate_LQR_explicit_batch(lqr_solver, x0, N, n_samples)  # Run simulation
            costs.append(cost)
        stderr = np.std(costs)  # Compute the standard error from the repeated runs
        errors.append(stderr)

        if i == 0:
            ref = stderr * math.sqrt(n_samples)  # Reference value for the theoretical error at the smallest sample size
        theory.append(ref / math.sqrt(n_samples))  # Theoretical error scales as 1/sqrt(n_samples)

        print(f"samples = {n_samples:6d}, StdErr = {stderr:.5f}")

    # Plot empirical standard error and theoretical error reference on a log-log scale
    plt.figure()
    plt.loglog(sample_list, errors, marker='o', label='Empirical Std. Error')
    plt.loglog(sample_list, theory, linestyle='--', label='1/sqrt(N) Reference')
    plt.xlabel('Number of Monte Carlo Samples')
    plt.ylabel('Standard Error')
    plt.title('Convergence vs Sample Size')
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.show()

    return sample_list, errors

def prepare_lqr_solver():
    """
    Set up LQRSolver with example matrices from Figure 1.
    """
    # Define system matrices and parameters
    H = np.array([[0.5, 0.5],
                  [0.0, 0.5]])
    M = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    sigma = 0.5 * np.eye(2)
    C = np.array([[1.0, 0.1],
                  [0.1, 1.0]])
    D = 0.1 * np.array([[1.0, 0.1],
                        [0.1, 1.0]])
    R = 10.0 * np.array([[1.0, 0.3],
                         [0.3, 1.0]])
    T = 0.5
    time_grid = np.linspace(0, T, 1000)  # Create a time grid from 0 to T with 1000 points

    # Instantiate the LQRSolver with the defined parameters
    lqr_solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    lqr_solver._solve_riccati_ode()  # Solve the Riccati ODE to obtain S(t) and the integral term
    return lqr_solver

def run_experiment_1_2_time_steps():
    """ Run time step convergence experiment. """
    lqr_solver = prepare_lqr_solver()  # Prepare the LQR solver with example parameters
    x0 = [1.0, 1.0]  # Define the initial state
    experiment_time_steps_stable(lqr_solver, x0)  # Run the time step experiment

def run_experiment_1_2_samples():
    """ Run sample size convergence experiment. """
    lqr_solver = prepare_lqr_solver()  # Prepare the LQR solver with example parameters
    x0 = [1.0, 1.0]  # Define the initial state
    experiment_samples_batchwise_stable(lqr_solver, x0)  # Run the sample size experiment

if __name__ == "__main__":
    # run the two experiments
    run_experiment_1_2_time_steps()
    run_experiment_1_2_samples() 

