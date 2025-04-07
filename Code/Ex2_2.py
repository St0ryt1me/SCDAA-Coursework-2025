import numpy as np
import torch
import matplotlib.pyplot as plt
from Ex2_1 import SoftLQRSolver  # Soft LQR implementation
from Ex1_1 import LQRSolver      # Strict LQR implementation

def simulate_explicit(lqr_solver, x0, N, n_samples, t0=0.0, use_soft=False, dW_all=None):
    """
    Simulate controlled trajectories using Euler-Maruyama discretization.
    Args:
        lqr_solver: instance of LQRSolver or SoftLQRSolver
        x0: initial state
        N: number of time steps
        n_samples: number of trajectories
        t0: initial time (default 0)
        use_soft: whether to use soft LQR control (stochastic)
        dW_all: optional pre-generated Brownian increments
    Returns:
        Array of shape (n_samples, N+1, state_dim) with trajectories
    """
    T = lqr_solver.T              # Terminal time
    tau = (T - t0) / N            # Compute time step size
    H = lqr_solver.H              # State dynamics matrix
    M = lqr_solver.M              # Control input matrix
    sigma = lqr_solver.sigma      # Noise (diffusion) matrix
    state_dim = len(x0)           # Determine the state dimension
    trajectories = np.zeros((n_samples, N+1, state_dim))  # Initialize array to store trajectories

    if dW_all is None:
        # Generate Brownian increments for all samples and time steps if not provided
        dW_all = np.sqrt(tau) * np.random.randn(n_samples, N, state_dim)

    for i in range(n_samples):
        X = np.array(x0, dtype=np.float32)  # Set initial state for current trajectory
        traj = [X.copy()]                   # Initialize trajectory list with the initial state
        t_curr = t0                         # Set current time to t0
        for n in range(N):
            S_t, _ = lqr_solver._get_S_at_time(t_curr)  # Get S(t) from solver at current time
            if use_soft:
                # Soft LQR: sample control from a Gaussian distribution
                mean = - lqr_solver.D_inv @ lqr_solver.M.T @ S_t @ X  # Compute mean control
                cov = (lqr_solver.reg_tau / lqr_solver.prior_gamma) * np.eye(state_dim)  # Covariance for control noise
                a = np.random.multivariate_normal(mean, cov)  # Sample control action stochastically
            else:
                # Strict LQR: use deterministic optimal control
                a = - lqr_solver.D_inv @ lqr_solver.M.T @ S_t @ X  # Compute optimal control action
            drift = H @ X + M @ a             # Compute drift term from dynamics and control
            dW = dW_all[i, n, :]              # Get Brownian increment for current step
            X = X + tau * drift + sigma @ dW   # Update state using Euler-Maruyama step
            t_curr += tau                   # Increment current time by tau
            traj.append(X.copy())           # Append updated state to trajectory list
        trajectories[i,:,:] = np.array(traj)  # Store complete trajectory for current sample
    return trajectories  # Return array of simulated trajectories

def plot_trajectories(traj_list, labels, title):
    """
    Plot averaged trajectories from multiple simulations.
    Args:
        traj_list: list of trajectory arrays, each of shape (n_samples, N+1, 2)
        labels: list of strings for legend
        title: plot title
    """
    plt.figure()  # Create a new figure for plotting
    for traj, lab in zip(traj_list, labels):
        avg_path = traj.mean(axis=0)  # Compute average trajectory over all samples
        plt.plot(avg_path[:,0], avg_path[:,1], marker='o', markersize=1, label=lab)  # Plot averaged path
    plt.xlabel("x1")   # Label for x-axis
    plt.ylabel("x2")   # Label for y-axis
    plt.title(title)   # Set the plot title
    plt.legend()       # Display legend
    plt.grid(True)     # Enable grid on the plot
    plt.show()         # Show the plot

if __name__ == "__main__":
    np.random.seed(42)
    # Setup problem parameters (refer to Figure 1)
    H = np.array([[0.5, 0.5],
                  [0.0, 0.5]])  # State dynamics matrix
    M = np.array([[1.0, 1.0],
                  [0.0, 1.0]])  # Control influence matrix
    sigma = 0.5 * np.eye(2)     # Noise covariance matrix (scaled identity)
    C = np.array([[1.0, 0.1],
                  [0.1, 1.0]])  # State cost matrix
    D = 0.1 * np.array([[1.0, 0.1],
                        [0.1, 1.0]])  # Control cost matrix (scaled)
    R = 10.0 * np.array([[1.0, 0.3],
                         [0.3, 1.0]])  # Terminal cost matrix
    T = 0.5                     # Terminal time
    time_grid = np.linspace(0, T, 1001)  # Create time grid from 0 to T with 1001 points

    # Instantiate strict LQR solver and solve its Riccati ODE
    strict_solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    strict_solver._solve_riccati_ode()

    # Instantiate soft LQR solver and solve its Riccati ODE
    soft_solver = SoftLQRSolver(H, M, sigma, C, D, R, T, time_grid, reg_tau=0.1, prior_gamma=10)
    soft_solver.solve_Riccati_ODE()

    # Define 4 initial states for simulation
    x0_list = [[2, 2], [2, -2], [-2, -2], [-2, 2]]
    n_samples = 1   # Use a small sample size for trajectory visualization
    N = 1000        # Number of time steps for simulation

    for x0 in x0_list:
        # Simulate trajectories using strict LQR (deterministic control)
        traj_strict = simulate_explicit(strict_solver, x0, N, n_samples, use_soft=False)
        # Simulate trajectories using soft LQR (stochastic control)
        traj_soft = simulate_explicit(soft_solver, x0, N, n_samples, use_soft=True)
        # Plot averaged trajectories for both methods with appropriate labels
        plot_trajectories([traj_strict, traj_soft],
                          labels=["Strict LQR", "Soft LQR"],
                          title=f"Trajectories from initial state {x0}")
