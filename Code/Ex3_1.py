import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------------
# Critic Network Definition
# ------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512):
        """
        Initialize the CriticNetwork.

        Parameters:
            input_dim (int): Dimension of the input (concatenated time and state).
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(CriticNetwork, self).__init__()
        # Define a feed-forward neural network using a sequential container.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First linear layer transforms input to hidden dimension.
            nn.ReLU(),                         # ReLU activation introduces non-linearity.
            nn.Linear(hidden_dim, hidden_dim), # Second linear layer for deeper representation.
            nn.ReLU(),                         # Another ReLU activation.
            nn.Linear(hidden_dim, 1)            # Final layer outputs a single scalar value (v(t,x)).
        )

    def forward(self, t, x):
        """
        Forward pass of the CriticNetwork.

        Parameters:
            t (Tensor): Time tensor.
            x (Tensor): State tensor.

        Returns:
            Tensor: The estimated value v(t, x).
        """
        # Concatenate time and state tensors along the feature dimension.
        inp = torch.cat([t, x], dim=1)
        # Pass the concatenated tensor through the network to get the value estimate.
        return self.net(inp)

# ------------------------------------
# Simulate a trajectory with log_prob tracking
# ------------------------------------
def simulate_episode_with_logprobs(soft_solver, x0, N):
    """
    Simulate a trajectory and track the log probabilities.

    Parameters:
        soft_solver: An instance of SoftLQRSolver providing system parameters and functions.
        x0 (array-like): Initial state.
        N (int): Number of time steps in the simulation.

    Returns:
        tuple: (t_list, x_list, f_list, terminal_cost, log_probs)
            t_list (ndarray): Array of time steps.
            x_list (ndarray): Array of states at each time step.
            f_list (ndarray): Array of stage costs at each time step.
            terminal_cost (float): Cost at the terminal state.
            log_probs (ndarray): Array of log probabilities for the sampled actions.
    """
    T = soft_solver.T  # Total simulation time
    dt = T / N         # Time step size
    state_dim = len(x0)  # Determine the dimension of the state
    t, x = 0.0, np.array(x0, dtype=np.float32)  # Initialize time and state

    # Initialize lists to store time, state, cost, and log probabilities.
    t_list, x_list, f_list, log_probs = [t], [x.copy()], [], []

    # Simulate the trajectory for N time steps.
    for _ in range(N):
        # Get the solution S_t at current time t using soft_solver.
        S_t, _ = soft_solver._get_S_at_time(t)
        # Calculate the optimal mean control in a feedback form.
        mean = - soft_solver.D_inv @ soft_solver.M.T @ S_t @ x
        # Construct the covariance matrix for the multivariate normal distribution.
        cov = (soft_solver.reg_tau / soft_solver.prior_gamma) * np.eye(state_dim)

        # Sample an action from the multivariate normal distribution.
        a = np.random.multivariate_normal(mean, cov)
        # Compute the inverse of covariance and the difference between sampled action and mean.
        cov_inv = np.linalg.inv(cov)
        diff = a - mean
        # Compute the log probability of the sampled action.
        log_prob = -0.5 * state_dim * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * diff.T @ cov_inv @ diff
        log_probs.append(log_prob)

        # Compute stage cost including the regularization term with the log probability.
        stage_cost = x.T @ soft_solver.C @ x + a.T @ soft_solver.D @ a + soft_solver.reg_tau * log_prob
        f_list.append(stage_cost)

        # Compute the drift term for state dynamics.
        drift = soft_solver.H @ x + soft_solver.M @ a
        # Generate noise using the Euler-Maruyama scheme.
        dW = np.sqrt(dt) * np.random.randn(state_dim)
        # Update the state using drift and diffusion.
        x = x + dt * drift + soft_solver.sigma @ dW
        t += dt  # Increment time

        # Store the new time and state.
        t_list.append(t)
        x_list.append(x.copy())

    # Compute the terminal cost at the final state.
    terminal_cost = x.T @ soft_solver.R @ x
    return np.array(t_list), np.array(x_list), np.array(f_list), terminal_cost, np.array(log_probs)

# ------------------------------------
# Compute Bellman targets
# ------------------------------------
def compute_bellman_target(f_list, log_probs, terminal_cost, dt):
    """
    Compute the Bellman target (cumulative return) for each time step.

    Parameters:
        f_list (array-like): List of stage costs.
        log_probs (array-like): List of log probabilities of the actions.
        terminal_cost (float): Cost at the terminal state.
        dt (float): Time step size.

    Returns:
        list: Bellman target for each time step.
    """
    G_list = []
    # For each time step, compute the cumulative return from that point.
    for n in range(len(f_list)):
        G = terminal_cost  # Start with terminal cost
        # Accumulate cost backwards from time step n to the end.
        for k in reversed(range(n, len(f_list))):
            G = (f_list[k] + log_probs[k]) * dt + G
        G_list.append(G)
    return G_list

# ------------------------------------
# Train critic using Bellman targets
# ------------------------------------
def train_critic_bellman(soft_solver, critic_net, optimizer, num_epochs=50, N=100, batch_size=500):
    """
    Train the critic network using Bellman targets computed from simulated episodes.

    Parameters:
        soft_solver: An instance of SoftLQRSolver providing system parameters and functions.
        critic_net (nn.Module): The critic network to be trained.
        optimizer (Optimizer): Optimizer for updating the critic network.
        num_epochs (int): Number of training epochs.
        N (int): Number of time steps per episode.
        batch_size (int): Number of episodes per batch.

    Returns:
        list: Loss values for each epoch.
    """
    losses = []
    # Iterate over the specified number of epochs.
    for epoch in range(num_epochs):
        all_t, all_x, all_G = [], [], []

        # For each epoch, simulate a batch of episodes.
        for _ in range(batch_size):
            # Sample an initial state uniformly from [-3, 3] for each state dimension.
            x0 = np.random.uniform(-3, 3, size=(2,))
            # Simulate an episode using the current soft_solver and initial state.
            t_list, x_list, f_list, terminal_cost, log_probs = simulate_episode_with_logprobs(soft_solver, x0, N)
            dt = soft_solver.T / N  # Compute time step size
            # Compute the Bellman target for the episode.
            G_list = compute_bellman_target(f_list, log_probs, terminal_cost, dt)

            # Exclude the terminal state from training data.
            all_t.append(t_list[:-1])
            all_x.append(x_list[:-1])
            all_G.append(G_list)

        # Concatenate the data from all episodes in the batch.
        t_tensor = torch.tensor(np.concatenate(all_t), dtype=torch.float32).unsqueeze(1)
        x_tensor = torch.tensor(np.concatenate(all_x), dtype=torch.float32)
        G_tensor = torch.tensor(np.concatenate(all_G), dtype=torch.float32).unsqueeze(1)

        # Predict the values using the critic network.
        v_pred = critic_net(t_tensor, x_tensor)
        # Compute the loss as the sum of squared differences.
        loss = torch.sum((v_pred - G_tensor) ** 2)

        # Perform backpropagation and update network parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        # Print the loss every 10 epochs for monitoring.
        if epoch % 5 == 0:
            print(f"[Bellman Critic] Epoch {epoch}, Batch Loss: {loss.item():.4e}")
    return losses

# ------------------------------------
# Main entry point
# ------------------------------------
if __name__ == "__main__":
    # Import SoftLQRSolver from Ex2_1 (ensure this class is implemented in Ex2_1.py)
    from Ex2_1 import SoftLQRSolver

    # Define system matrices and parameters.
    H = np.array([[0.5, 0.5], [0.0, 0.5]], dtype=np.float32)
    M = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    sigma = 0.5 * np.eye(2, dtype=np.float32)
    C = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32)
    D = np.eye(2, dtype=np.float32)
    R = 10.0 * np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float32)

    T = 0.5  # Total simulation time
    time_grid = np.linspace(0, T, 1001)  # Discretized time grid for the solver
    reg_tau = 0.5  # Regularization parameter (τ)
    prior_gamma = 1.0  # Prior scaling factor (γ)

    # Set random seeds for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize the SoftLQRSolver with system parameters.
    soft_solver = SoftLQRSolver(H, M, sigma, C, D, R, T, time_grid, reg_tau, prior_gamma)
    # Solve the Riccati ODE to obtain the value function solution.
    soft_solver.solve_Riccati_ODE()

    # Initialize the Critic network and its optimizer.
    critic_net = CriticNetwork()
    optimizer = optim.Adam(critic_net.parameters(), lr=1e-3)

    print("Starting Bellman Critic training with 500 episodes per batch...")
    # Train the critic network using the Bellman target method.
    losses = train_critic_bellman(soft_solver, critic_net, optimizer, num_epochs=50, N=100, batch_size=500)
    # After training, conduct rigorous testing:
    # Construct the test grid: t in {0, 1/6, 2/6, 1/2} and x in [-3,3]x[-3,3]
    t_set = np.array([0.0, 1/6, 2/6, 0.5], dtype=np.float32)
    N_points = 25  # 25 points in each dimension
    x_range = np.linspace(-3, 3, N_points, dtype=np.float32)
    X1, X2 = np.meshgrid(x_range, x_range)
    X1 = X1.flatten()
    X2 = X2.flatten()
    
    # For each t, compute the predicted value from the Critic and the analytic value
    max_error = 0.0
    for t_val in t_set:
        t_tensor = torch.tensor([t_val]*len(X1), dtype=torch.float32).unsqueeze(1)
        x_tensor = torch.tensor(np.stack([X1, X2], axis=1), dtype=torch.float32)
        with torch.no_grad():
            v_pred = critic_net(t_tensor, x_tensor)
        v_pred = v_pred.squeeze().numpy()
        # The analytic value is computed using soft_solver.compute_value_function
        t_tensor_eval = torch.tensor([t_val]*len(X1), dtype=torch.float32).unsqueeze(1)
        v_analytic = soft_solver.compute_value_function(t_tensor_eval, x_tensor).numpy()
        error = np.abs(v_pred - v_analytic)
        max_error = max(max_error, error.max())
        print(f"t = {t_val:.3f}, max error = {error.max():.4e}")
    
    print(f"Global maximum error = {max_error:.4e}")
    # Plot the training loss over epochs.
    plt.plot(losses)
    plt.title("Critic Bellman Loss (500 episodes per epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Sum over batch)")
    plt.grid(True)
    plt.show()

# Import necessary tools for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D

def plot_value_surface(critic_net, soft_solver, t_fixed=0.0, x_range=np.linspace(-3, 3, 50)):
    """
    Plot the predicted critic value surface alongside the analytic value function at a fixed time.

    Parameters:
        critic_net (nn.Module): The trained critic network.
        soft_solver: An instance of SoftLQRSolver used to compute the analytic value function.
        t_fixed (float): The fixed time at which to evaluate the value functions.
        x_range (ndarray): Range of state values for plotting.

    Returns:
        None
    """
    # Create a mesh grid for the state space.
    X1, X2 = np.meshgrid(x_range, x_range)
    # Flatten the grid to form input vectors.
    X_flat = np.stack([X1.flatten(), X2.flatten()], axis=1)
    # Create a tensor for the fixed time, matching the number of state samples.
    t_tensor = torch.full((X_flat.shape[0], 1), t_fixed, dtype=torch.float32)
    # Convert the flattened state grid to a tensor.
    x_tensor = torch.tensor(X_flat, dtype=torch.float32)

    with torch.no_grad():
        # Get predicted values from the critic network.
        v_pred = critic_net(t_tensor, x_tensor).squeeze().numpy()
    # Compute the analytic value function using the soft solver.
    v_true = soft_solver.compute_value_function(t_tensor, x_tensor).numpy()

    # Reshape the predicted and true values to match the mesh grid shape.
    Vp = v_pred.reshape(X1.shape)
    Vt = v_true.reshape(X1.shape)

    # Create a figure for plotting the two surfaces side by side.
    fig = plt.figure(figsize=(12, 5))

    # Plot the critic-predicted value surface.
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X1, X2, Vp, cmap='viridis')
    ax1.set_title(f"Critic Predicted Value at t={t_fixed:.2f}")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("v(t,x)")

    # Plot the analytic value surface.
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X1, X2, Vt, cmap='plasma')
    ax2.set_title(f"Analytic Value at t={t_fixed:.2f}")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("v(t,x)")

    plt.tight_layout()
    plt.show()

# Call the plotting function to visualize value surfaces at a fixed time (t = 0).
plot_value_surface(critic_net, soft_solver, t_fixed=0)
