import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Ex2_1 import SoftLQRSolver

# -----------------------------
# Actor Network Definition
# -----------------------------
class ActorNetwork(nn.Module):
    """
    Actor network for stochastic policy: maps (t, x) -> (mean, std)
    Inputs:
        t: time tensor of shape (batch, 1)
        x: state tensor of shape (batch, 2)
    Output:
        mean: mean of Gaussian policy (batch, 2)
        std: standard deviation (shared, learnable, shape (2,))
    """
    def __init__(self, state_dim=2, hidden_size=256):
        super(ActorNetwork, self).__init__()
        # First fully connected layer: input dimension is state_dim + 1 (for time)
        self.fc1 = nn.Linear(state_dim + 1, hidden_size)
        # Second fully connected layer for further feature extraction
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer for computing the mean of the action distribution
        self.fc_mean = nn.Linear(hidden_size, state_dim)
        # Learnable parameter for the log standard deviation (shared among actions)
        self.fc_log_std = nn.Parameter(torch.zeros(state_dim))  # Shared log std
        self.relu = nn.ReLU()

    def forward(self, t, x):
        # Concatenate time and state tensors along the feature dimension.
        inp = torch.cat([t, x], dim=1)
        # Pass through the first fully connected layer and apply ReLU activation.
        h = self.relu(self.fc1(inp))
        # Pass through the second fully connected layer and apply ReLU activation.
        h = self.relu(self.fc2(h))
        # Compute the mean of the action distribution.
        mean = self.fc_mean(h)
        # Calculate the standard deviation by exponentiating the shared log standard deviation.
        log_std = self.fc_log_std
        std = torch.exp(log_std)
        return mean, std

# -----------------------------
# Simulate a trajectory from the actor policy
# -----------------------------
def simulate_episode(actor_net, H, M, sigma, C, D, R, tau, x0, T, N):
    """
    Simulate one trajectory using current actor policy.
    
    Parameters:
        actor_net: The actor network used to generate actions.
        H, M, sigma, C, D, R: System dynamics and cost parameters.
        tau: Entropy regularization coefficient.
        x0: Initial state as a list or array.
        T: Total simulation time.
        N: Number of time steps.
    
    Returns:
        t_list: List of time points.
        x_list: List of state vectors.
        logp_list: List of log probabilities for each action.
        f_list: List of computed costs (including entropy term) at each step.
    """
    dt = T / N  # Time step size
    t = 0.0
    x = torch.tensor(x0, dtype=torch.float32)  # Convert initial state to tensor
    t_list, x_list, logp_list, f_list = [t], [x.clone()], [], []

    # Convert matrices to torch tensors for compatibility in computations.
    H_tensor = torch.tensor(H, dtype=torch.float32)
    M_tensor = torch.tensor(M, dtype=torch.float32)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

    for _ in range(N):
        # Create a tensor for the current time (as a batch of size 1).
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        # Add batch dimension to the state tensor.
        x_tensor = x.unsqueeze(0)
        # Obtain the mean and standard deviation from the actor network.
        mean, std = actor_net(t_tensor, x_tensor)
        # Define a normal distribution based on the network's output.
        dist = torch.distributions.Normal(mean, std)
        # Sample an action from the distribution.
        a = dist.sample().squeeze(0)
        # Compute the log probability of the sampled action.
        logp = dist.log_prob(a).sum()  # Sum log probabilities over action dimensions.
        logp_list.append(logp)

        # Compute the instantaneous cost including the entropy regularization term.
        f = (x @ torch.tensor(C) @ x + a @ torch.tensor(D) @ a).item() + tau * logp.item()
        f_list.append(f)

        # Euler-Maruyama update for stochastic differential equation dynamics.
        drift = H_tensor @ x + M_tensor @ a
        dW = torch.tensor(np.sqrt(dt) * np.random.randn(len(x)), dtype=torch.float32)  # Noise term
        x = x + dt * drift + sigma_tensor @ dW
        t += dt  # Increment time by one time step
        t_list.append(t)
        x_list.append(x.clone())

    return t_list, x_list, logp_list, f_list

# -----------------------------
# Actor Training (Policy Gradient)
# -----------------------------
def train_actor_epoch(actor_net, optimizer, H, M, sigma, C, D, R, tau,
                      T=0.5, N=100, num_epochs=50, batch_size=500):
    """
    Train the actor network using actor-only policy gradient.
    Compute delta V + (stage cost + tau * log_prob) and use it for the gradient update.
    
    Parameters:
        actor_net: The actor network to train.
        optimizer: Optimizer for updating network parameters.
        H, M, sigma, C, D, R: System dynamics and cost parameters.
        tau: Entropy regularization coefficient.
        T: Total simulation time.
        N: Number of time steps per episode.
        num_epochs: Total number of training epochs.
        batch_size: Number of episodes per epoch.
    
    Returns:
        abs_losses: List containing the absolute policy gradient loss for each epoch.
    """
    abs_losses = []

    # Create exact value function using soft LQR (used as a baseline for policy gradient)
    soft_solver = SoftLQRSolver(H, M, sigma, C, D, R, T,
                                 time_grid=np.linspace(0, T, 1001),
                                 reg_tau=tau, prior_gamma=1.0)
    soft_solver.solve_Riccati_ODE()  # Solve the Riccati differential equation

    for epoch in range(num_epochs):
        acc_loss = 0.0  # Accumulate the gradient loss over episodes
        loss_500 = 0.0  # Accumulate absolute loss (for logging)
        for _ in range(batch_size):
            # Randomly initialize state x0 within the range [-2, 2] for each state dimension.
            x0 = np.random.uniform(-2, 2, size=(2,))
            # Simulate one episode with the current actor policy.
            t_list, x_list, logp_list, f_list = simulate_episode(actor_net, H, M, sigma, C, D, R, tau, x0, T, N)
            dt = T / N

            delta_v_list = []
            for n in range(N):
                # Estimate the value difference: delta v = v(t_{n+1}, x_{n+1}) - v(t_n, x_n)
                t_n = t_list[n]
                x_n = x_list[n].numpy()
                S_t_n, b_t_n = soft_solver._get_S_at_time(t_n)
                v_n = x_n @ S_t_n @ x_n + b_t_n

                t_n1 = t_list[n+1]
                x_n1 = x_list[n+1].numpy()
                S_t_n1, b_t_n1 = soft_solver._get_S_at_time(t_n1)
                v_n1 = x_n1 @ S_t_n1 @ x_n1 + b_t_n1

                delta_v = v_n1 - v_n
                delta_v_list.append(delta_v)

            # Compute the policy gradient: sum_t log_pi(a_t) * (delta_v + cost_t)
            pg_sum = 0.0
            for n in range(N):
                term = delta_v_list[n] + (f_list[n] + tau * logp_list[n]) * dt
                pg_sum += logp_list[n] * term

            acc_loss += pg_sum
            loss_500 += abs(pg_sum)

        optimizer.zero_grad()  # Clear previous gradients
        acc_loss.backward()  # Backpropagate the accumulated loss
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)  # Clip gradients to prevent exploding gradients
        optimizer.step()  # Update the actor network parameters

        abs_loss_value = abs(loss_500.item())
        abs_losses.append(abs_loss_value)

        # Print training progress at the first epoch and every 5 epochs thereafter.
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"[Actor PG] >>> Epoch {epoch + 1}: |PG Loss| = {loss_500.item():.2f}")

    return abs_losses

# -----------------------------
# Plot Actor Mean vs Optimal Control (u1 only)
# -----------------------------
def plot_actor_vs_optimal(actor_net, soft_solver, D, M, T=0.5):
    """
    Compare the actor policy's mean action with the analytic optimal control on the u1 control input.
    
    This function generates two 3D surface plots: one for the actor's control and one for the optimal control.
    """
    x1_vals = np.linspace(-2, 2, 30)
    x2_vals = np.linspace(-2, 2, 30)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    U1_actor = np.zeros_like(X1)
    U1_opt = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]], dtype=np.float32)
            x_tensor = torch.tensor(x).unsqueeze(0)
            t_tensor = torch.tensor([[0.0]], dtype=torch.float32)

            with torch.no_grad():
                mean, _ = actor_net(t_tensor, x_tensor)
                a_actor = mean.numpy().squeeze()
            U1_actor[i, j] = a_actor[0]

            # Calculate optimal control using the soft LQR solution.
            S0, _ = soft_solver._get_S_at_time(0.0)
            a_opt = -np.linalg.inv(D) @ M.T @ S0 @ x
            U1_opt[i, j] = a_opt[0]

    # Create a figure with two 3D subplots for comparison.
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1, X2, U1_actor, cmap='viridis')
    ax1.set_title("Actor Control $u_1$ (mean)")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$u_1$")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X1, X2, U1_opt, cmap='plasma')
    ax2.set_title("Optimal Control $u_1$")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_zlabel("$u_1$")

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Set random seeds for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)

    # Define system dynamics and cost matrices.
    H = np.array([[0.5, 0.5], [0.0, 0.5]], dtype=np.float32)
    M = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    sigma = 0.5 * np.eye(2, dtype=np.float32)
    C = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32)
    D = np.eye(2, dtype=np.float32)
    R = 10.0 * np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float32)
    T = 0.5
    tau = 0.5

    # Initialize the actor network and the optimizer.
    actor_net = ActorNetwork()
    optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)

    print("Starting Actor-only policy training ...")
    abs_losses = train_actor_epoch(actor_net, optimizer, H, M, sigma, C, D, R, tau,
                                   T=T, N=100, num_epochs=50, batch_size=500)

    # Plot the absolute policy gradient loss per epoch.
    plt.plot(abs_losses)
    plt.title("Absolute Policy Gradient Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("|PG Loss|")
    plt.grid(True)
    plt.show()

    # Compute the soft LQR solution for the optimal control.
    soft_solver = SoftLQRSolver(H, M, sigma, C, D, R, T,
                                 time_grid=np.linspace(0, T, 1001),
                                 reg_tau=tau, prior_gamma=1.0)
    soft_solver.solve_Riccati_ODE()
    # Visualize and compare the actor's control surface with the optimal control surface.
    plot_actor_vs_optimal(actor_net, soft_solver, D, M, T)
