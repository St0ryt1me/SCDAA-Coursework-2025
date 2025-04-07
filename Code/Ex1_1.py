import numpy as np
import torch
from scipy.integrate import solve_ivp  # Import ODE solver for integrating the Riccati ODE
from scipy.integrate import cumulative_trapezoid as cumtrapz   # Import function to compute cumulative trapezoidal integration

# Define the LQRSolver class to solve the Linear Quadratic Regulator problem.
class LQRSolver:
    def __init__(self, H, M, sigma, C, D, R, T, time_grid):
        """
        Initialize LQR solver.
        Args:
            H, M, sigma, C, D, R: Matrices defining the LQR problem.
            T: Terminal time (scalar).
            time_grid: 1D numpy array for time discretization on [t0, T].
        """
        self.H = np.array(H)             # System matrix for the state dynamics
        self.M = np.array(M)             # Control input matrix
        self.sigma = np.array(sigma)       # Diffusion matrix (for noise)
        self.C = np.array(C)             # State cost matrix
        self.D = np.array(D)             # Control cost matrix
        self.R = np.array(R)             # Terminal cost matrix
        self.T = T                       # Terminal time
        self.time_grid = np.array(time_grid)  # Discretized time grid for the solution

        self.D_inv = np.linalg.inv(self.D)  # Precompute the inverse of D for efficiency

        self.S_solution = None  # This will store the computed S(t) matrices on the time grid
        self._integral_term = None  # This will store the precomputed integral term: ∫ₜᵀ tr(sigma sigma^T S(r)) dr

    def _riccati_ode(self, t, S_flat):
        """
        Right-hand side of Riccati ODE (vector form).
        Note: This is a terminal condition problem S(T) = R, so integration is backward in time.
        ODE: S'(t) = S M D^{-1} M^T S - H^T S - S H - C.
        """
        S = S_flat.reshape(self.R.shape)  # Reshape the flat vector back into matrix form
        term1 = S @ self.M @ self.D_inv @ self.M.T @ S  # Compute quadratic term S M D^{-1} M^T S
        term2 = self.H.T @ S + S @ self.H  # Compute the symmetric term H^T S + S H
        dSdt = term1 - term2 - self.C     # The derivative dS/dt per the Riccati ODE
        return dSdt.flatten()             # Return the derivative as a flattened vector

    def _solve_riccati_ode(self):
        """
        Solve Riccati ODE and interpolate solution over time grid.
        Backward solve from T to t0.
        """
        t0 = self.time_grid[0]  # Starting time of the integration (t0)
        # Solve the Riccati ODE backward in time from T to t0 using RK45 method
        sol = solve_ivp(fun=self._riccati_ode, t_span=(self.T, t0), y0=self.R.flatten(),
                        t_eval=self.time_grid[::-1], method='RK45', rtol=1e-6, atol=1e-8)
        if not sol.success:
            raise RuntimeError("Riccati ODE solver failed.")

        # Reverse the solution so that the time grid is in ascending order
        S_all = sol.y.T[::-1]
        # Reshape the solution to have shape (num_time_steps, n, n)
        self.S_solution = S_all.reshape(-1, self.R.shape[0], self.R.shape[1])

        # Precompute the integral term: ∫ₜᵀ tr(sigma sigma^T S(r)) dr using the trapezoidal rule
        sigma_sigmaT = self.sigma @ self.sigma.T  # Compute sigma*sigma^T once since it's constant
        # For each S in the solution, compute the trace of (sigma*sigma^T @ S)
        trace_vals = np.array([np.trace(sigma_sigmaT @ S) for S in self.S_solution])
        # Compute the cumulative integral using cumtrapz (flip arrays to integrate backward)
        cumulative_integral = -np.flip(cumtrapz(np.flip(trace_vals), np.flip(self.time_grid), initial=0))
        self._integral_term = cumulative_integral

    def _get_S_at_time(self, t_val):
        """
        Retrieve approximate S(t) from precomputed grid by nearest (<= t).
        """
        # Find the index of the nearest time in the grid that is less than or equal to t_val
        idx = np.searchsorted(self.time_grid, t_val, side='right') - 1
        idx = np.clip(idx, 0, len(self.time_grid) - 1)  # Ensure index is within bounds
        return self.S_solution[idx], self._integral_term[idx]  # Return both S and the integral term at that time

    def value_function(self, t, x):
        """
        Compute value function v(t, x) = x^T S(t) x + ∫ₜᵀ tr(sigma sigma^T S(r)) dr.

        Args:
            t: torch 1D tensor (times)
            x: torch 2D tensor (states), shape (N, 2)
        Returns:
            torch 1D tensor of value function at each (t, x) pair.
        """
        # Convert torch tensors to numpy arrays for indexing
        t_np = t.numpy() if isinstance(t, torch.Tensor) else t
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x

        # Find corresponding indices in the time grid for each t value
        indices = np.clip(np.searchsorted(self.time_grid, t_np, side='right') - 1, 0, len(self.time_grid)-1)
        S_t = self.S_solution[indices]         # Get S(t) for each index
        integral_t = self._integral_term[indices]  # Get corresponding integral term

        # Compute quadratic cost term: x^T S(t) x for each sample
        quad_terms = np.einsum('ni,nij,nj->n', x_np, S_t, x_np)
        # Total value is quadratic term plus the integral term
        values = quad_terms + integral_t
        return torch.tensor(values, dtype=torch.float32)

    def optimal_control(self, t_tensor, x_tensor):
        """
        Compute optimal control a(t,x) = -D^{-1} M^T S(t) x
        for each (t, x) pair.
        """
        # Detach time tensor and convert to numpy for indexing
        t_vals = t_tensor.detach().cpu().numpy()
        n_samples = len(t_vals)
        x_dim = x_tensor.shape[1]
        controls_np = np.zeros((n_samples, x_dim), dtype=np.float32)

        # Loop over each sample to compute control using precomputed S(t)
        for i, t_val in enumerate(t_vals):
            S_t, _ = self._get_S_at_time(t_val)  # Retrieve S(t) for the current time
            x = x_tensor[i].detach().cpu().numpy()  # Get the state vector
            a_val = - self.D_inv @ self.M.T @ S_t @ x  # Compute the optimal control
            controls_np[i] = a_val

        return torch.from_numpy(controls_np)

# =============================
# Example usage
if __name__ == "__main__":
    # Define example matrices from SCDAA coursework Figure 1
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
    time_grid = np.linspace(0, T, 101)  # Create a time grid with 101 points between 0 and T

    # Instantiate the LQR solver with the specified system parameters and time grid.
    solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    solver._solve_riccati_ode()  # Solve the Riccati ODE to obtain S(t) and the integral term

    # Test the value function and optimal control functions with example inputs.
    t_test = torch.tensor([0.0], dtype=torch.float32)       # Test time (t = 0)
    x_test = torch.tensor([[1.0, 1.0]], dtype=torch.float32)  # Test state vector

    v_vals = solver.value_function(t_test, x_test)  # Compute the value function at the test point
    a_vals = solver.optimal_control(t_test, x_test)   # Compute the optimal control at the test point

    # Print the results to verify the computations.
    print("Value function at test points:", v_vals)
    print("Optimal control at test points:", a_vals)
