import numpy as np
import torch
from scipy.integrate import solve_ivp, cumtrapz  # Import ODE solver and cumulative trapezoidal integration function

class SoftLQRSolver:
    def __init__(self, H, M, sigma, C, D, R, T, time_grid, reg_tau, prior_gamma):
        """
        Initialize the soft LQR solver (Exercise 2.1, Part 1)
        Args:
          H, M, sigma, C, D, R: matrices (as numpy arrays or convertible)
          T: terminal time
          time_grid: 1D numpy array over [t0, T]
          reg_tau: entropy regularization strength (tau)
          prior_gamma: variance parameter for prior Gaussian
        """
        self.H = np.array(H)                # System dynamics matrix
        self.M = np.array(M)                # Control input matrix
        self.sigma = np.array(sigma)          # Diffusion/noise matrix
        self.C = np.array(C)                # State cost matrix
        self.D = np.array(D)                # Control cost matrix
        self.R = np.array(R)                # Terminal cost matrix
        self.T = T                          # Terminal time
        self.time_grid = np.array(time_grid)  # Discretized time grid over [t0, T]
        self.D_inv = np.linalg.inv(self.D)  # Precompute the inverse of D for later use

        # Soft LQR specific parameters
        self.reg_tau = reg_tau              # Regularization parameter for entropy term
        self.prior_gamma = prior_gamma      # Variance parameter for the prior Gaussian

        self.S_solution = None              # To store the computed S(t) matrices over the time grid
        self._integral_term = None          # To store the precomputed integral: ∫_t^T tr(sigma sigma^T S(r)) dr

    def _riccati_ode(self, t, S_flat):
        """
        Right-hand side of Riccati ODE for soft LQR.
        ODE: S'(t) = S M D⁻¹ Mᵀ S - Hᵀ S - S H - C - (1/\gamma) M D⁻¹ Mᵀ
        """
        S = S_flat.reshape(self.R.shape)   # Reshape the flat vector back to the matrix form of S
        term1 = S @ self.M @ self.D_inv @ self.M.T @ S  # Compute quadratic term: S M D⁻¹ Mᵀ S
        term2 = self.H.T @ S + S @ self.H     # Compute symmetric term: Hᵀ S + S H
        term3 = 1/self.prior_gamma * self.M @ self.D_inv @ self.M.T  # Compute additional constant term
        dSdt = term1 - term2 - self.C - term3  # Compute derivative of S using the Riccati ODE
        return dSdt.flatten()                # Return the derivative as a flattened vector for the ODE solver

    def solve_Riccati_ODE(self):
        """
        Solve Riccati ODE and store S(t) and integral term over time grid.
        """
        t0 = self.time_grid[0]  # Starting time of the grid (lowest time value)
        # Solve the ODE backward in time from T to t0 using RK45 method
        sol = solve_ivp(fun=self._riccati_ode, t_span=(self.T, t0), y0=self.R.flatten(),
                        t_eval=self.time_grid[::-1], method='RK45', rtol=1e-6, atol=1e-8)
        if not sol.success:
            raise RuntimeError("Riccati ODE solver failed.")

        S_all = sol.y.T[::-1]  # Reverse the solution to have ascending time order
        self.S_solution = S_all.reshape(-1, self.R.shape[0], self.R.shape[1])  # Reshape to [num_times, n, n]

        # Precompute the integral term ∫_t^T tr(sigma sigma^T S(r)) dr for each time point
        sigma_sigmaT = self.sigma @ self.sigma.T  # Compute sigma * sigma^T once (constant)
        trace_vals = np.array([np.trace(sigma_sigmaT @ S) for S in self.S_solution])  # Compute trace for each S
        cumulative_integral = np.flip(cumtrapz(np.flip(trace_vals), np.flip(self.time_grid), initial=0))
        self._integral_term = cumulative_integral  # Store the computed integral term

    def _get_S_at_time(self, t_val):
        """
        Return S(t) and integral term at time t_val using nearest left grid point.
        """
        # Find the index of the largest time in the grid that is <= t_val
        idx = np.searchsorted(self.time_grid, t_val, side='right') - 1
        idx = np.clip(idx, 0, len(self.time_grid) - 1)  # Ensure the index is within valid range
        return self.S_solution[idx], self._integral_term[idx]  # Return S(t) and corresponding integral term

    def compute_value_function(self, t_tensor, x_tensor):
        """
        Compute v(t,x) = x^T S(t) x + \int_t^T tr(sigma sigma^T S(r)) dr
        """
        t_vals = t_tensor.detach().cpu().numpy()  # Convert torch tensor to numpy array for indexing
        results = []  # List to accumulate computed value function for each (t, x)
        for i, t_val in enumerate(t_vals):
            S_t, integral_val = self._get_S_at_time(t_val)  # Get S(t) and integral at time t_val
            x = x_tensor[i].detach().cpu().numpy()          # Get state vector for sample i
            quad = x.T @ S_t @ x  # Compute quadratic term x^T S(t) x
            results.append(quad + integral_val)  # Sum quadratic term and integral term
        return torch.tensor(np.array(results), dtype=torch.float32)  # Return value function as a torch tensor

    def compute_optimal_control(self, t_tensor, x_tensor):
        """
        Sample control from Gaussian distribution in soft LQR.
        Mean:   -D⁻¹ Mᵀ S(t) x
        Covar:  (tau / gamma) * D⁻¹
        Return sampled actions from N(mean, cov).
        """
        t_vals = t_tensor.detach().cpu().numpy()  # Convert time tensor to numpy array
        controls = []  # List to accumulate sampled control actions
        for i, t_val in enumerate(t_vals):
            S_t, _ = self._get_S_at_time(t_val)  # Get S(t) at the given time
            x = x_tensor[i].detach().cpu().numpy()  # Get corresponding state vector
            mean = - self.D_inv @ self.M.T @ S_t @ x  # Compute the mean control action
            cov = (self.reg_tau / self.prior_gamma) * self.D_inv  # Compute covariance matrix for control
            a_sample = np.random.multivariate_normal(mean, cov)  # Sample control action from Gaussian
            controls.append(a_sample)
        return torch.tensor(np.array(controls), dtype=torch.float32)  # Return sampled controls as a torch tensor
