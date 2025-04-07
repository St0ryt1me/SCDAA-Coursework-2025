import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import SoftLQRSolver from Ex2_1 (ensure that the class is implemented in Ex2_1.py)
from Ex2_1 import SoftLQRSolver

# -------------------------
# 1. Define the Actor Network
# -------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim=2, hidden_size=512, device="cpu"):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, hidden_size)  # Input is (t, x)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, state_dim)  # Output mean (no tanh needed)
        self.fc_log_std = nn.Parameter(torch.zeros(state_dim))  # Fixed variance (like fixed covariance in the paper)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)  # Concatenate time and state along the feature dimension.
        h = self.relu(self.fc1(inp))     # First hidden layer with ReLU activation.
        h = self.relu(self.fc2(h))       # Second hidden layer with ReLU activation.
        mean = self.fc_mean(h)           # Compute the mean of the action distribution.
        log_std = self.fc_log_std         # Fixed variance parameter (as τΣ in the paper)
        std = torch.exp(log_std)          # Exponentiate to ensure positive standard deviation.
        return mean, std


# -------------------------
# 2. Define the Critic Network
# -------------------------
class CriticNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512):
        """
        Input: Concatenated time t and state x (3 dimensions), output: scalar v(t, x)
        """
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)  # Concatenate time and state.
        return self.net(inp)           # Compute the value function estimate.


# -------------------------
# 3. Sample an Actor-Critic Episode
# -------------------------
def simulate_ac_episode(soft_solver, actor_net, x0, N, dt):
    """
    Sample an episode:
      - Use the current Actor network to sample actions and record their log probabilities.
      - Record the state, action, log probability, and stage cost f_t = xᵀ C x + aᵀ D a at each time step.
      - Update the state using the Euler method: xₙ₊₁ = xₙ + dt*(H xₙ + M aₙ) + σ √(dt) ξ
      
    Returns:
      t_list: List of length N+1 with time points (floats)
      x_list: List of length N+1, each is a tensor with shape (state_dim,)
      a_list: List of length N, each is a tensor with shape (state_dim,)
      logp_list: List of length N, each is a scalar tensor (not detached)
      f_list: List of length N, each is the immediate cost (float)
    """
    t_list = []
    x_list = []
    a_list = []
    logp_list = []
    f_list = []

    t = 0.0
    state_dim = len(x0)
    x = torch.from_numpy(np.array(x0, dtype=np.float32))
    t_list.append(t)
    x_list.append(x.clone())
    
    for n in range(N):
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        x_tensor = x.unsqueeze(0)  # Shape: (1, state_dim)
        mean, std = actor_net(t_tensor, x_tensor)  # Get action distribution parameters from Actor.
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()  # Sample an action; shape: (1, state_dim)
        logp = dist.log_prob(a).sum(dim=1)  # Compute log probability of the sampled action (scalar, not detached).
        a = a.squeeze(0)   # Remove batch dimension.
        logp = logp.squeeze(0)  # Remove batch dimension.
        
        a_list.append(a)
        logp_list.append(logp)
        
        # Compute stage cost: f_t = xᵀ C x + aᵀ D a
        # Note: This cost does not include the entropy term; τ * ln p will be added later during gradient update.
        x_np = x.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        # Use soft_solver.reg_tau to represent τ, and logp.item() to obtain the numerical value of ln p (note that logp is negative)
        stage_cost = x_np.T @ soft_solver.C @ x_np + a_np.T @ soft_solver.D @ a_np
        f_list.append(stage_cost)

        # State update using the Euler method for stochastic differential equations.
        drift = torch.tensor(soft_solver.H, dtype=torch.float32) @ x + \
                torch.tensor(soft_solver.M, dtype=torch.float32) @ a
        noise = torch.tensor(np.sqrt(dt)*np.random.randn(state_dim), dtype=torch.float32)
        sigma = torch.tensor(soft_solver.sigma, dtype=torch.float32)
        x = x + dt * drift + sigma @ noise  # Update state with drift and noise.
        t = t + dt  # Increment time by dt.
        t_list.append(t)
        x_list.append(x.clone())
    
    return t_list, x_list, a_list, logp_list, f_list

# -------------------------
# 4. Compute Discrete Time Difference of v* (Using Critic Network Predictions)
# -------------------------
def compute_delta_v_from_critic(critic_net, t_list, x_list):
    """
    Use the Critic network to compute the predicted value v(t,x) at each time step, then compute:
      Δv_n = v(tₙ₊₁, xₙ₊₁) - v(tₙ, xₙ)
    Returns a list of length N.
    """
    v_list = []
    for t_val, x_val in zip(t_list, x_list):
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
        # x_val is already a tensor; add a batch dimension.
        x_tensor = x_val.unsqueeze(0)
        with torch.no_grad():
            v = critic_net(t_tensor, x_tensor)
        v_list.append(v.squeeze(0))
    delta_vs = []
    for n in range(len(v_list)-1):
        delta_vs.append(v_list[n+1].detach() - v_list[n].detach())
    return delta_vs

# -------------------------
# 5. Actor-Critic Offline Algorithm
# -------------------------
def train_actor_critic_epoch(actor_net, critic_net, actor_optimizer, critic_optimizer,
                             soft_solver, num_epochs=50, batch_size=500, N=100):
    """
    In each epoch, sample batch_size episodes and update the network parameters using the actor-critic method.
    The structure mimics that in wjz4.1.py, updating once per epoch and printing the results for the first epoch and every 5 epochs.
    
    Parameters:
      actor_net: The Actor network to be trained.
      critic_net: The Critic network for value estimation.
      actor_optimizer: Optimizer for the Actor network.
      critic_optimizer: Optimizer for the Critic network.
      soft_solver: An instance of SoftLQRSolver providing system parameters and solutions.
      num_epochs: Number of training epochs.
      batch_size: Number of episodes per epoch.
      N: Number of time steps per episode.
    
    Returns:
      actor_losses_abs: List of absolute Actor loss values per epoch.
      critic_losses: List of Critic loss values per epoch.
    """
    tau = soft_solver.reg_tau
    T = soft_solver.T
    dt = T / N

    actor_losses = []
    critic_losses = []
    actor_losses_abs = []
    for epoch in range(num_epochs):
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        loss_500 = 0.0
        for _ in range(batch_size):
            # Sample initial state uniformly from [-3, 3].
            x0 = np.random.uniform(-3, 3, size=(2,))
            t_list, x_list, a_list, logp_list, f_list = simulate_ac_episode(soft_solver, actor_net, x0, N, dt)

            # Compute Critic network estimates for each time step.
            v_list = []
            for t_val, x_val in zip(t_list, x_list):
                t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
                x_tensor = x_val.unsqueeze(0)
                v = critic_net(t_tensor, x_tensor)
                v_list.append(v.squeeze(0))

            # Compute the differences Δv for each time step.
            delta_vs = []
            for n in range(len(v_list) - 1):
                delta_vs.append(v_list[n+1].detach() - v_list[n].detach())

            # Accumulate Actor loss using the policy gradient.
            pg_sum = 0.0
            for n in range(N):
                delta_v = delta_vs[n]  # Use delta v from Critic prediction.
                term = delta_v + (f_list[n] + tau * logp_list[n]) * dt
                pg_sum += logp_list[n] * term
            actor_loss_sum += pg_sum
            loss_500 += abs(pg_sum)
            
            # Accumulate Critic loss.
            terminal_cost = x_list[-1].unsqueeze(0)
            terminal_cost_val = (terminal_cost @ torch.tensor(soft_solver.R, dtype=torch.float32) @ terminal_cost.t()).item()

            G = []
            cumulative = terminal_cost_val
            # Compute the cumulative return G backwards.
            for f_k, logp_k in reversed(list(zip(f_list, logp_list))):
                cumulative = (f_k + tau * logp_k.item()) * dt + cumulative
                G.insert(0, cumulative)
            G = np.array(G)

            t_tensor = torch.tensor(t_list[:-1], dtype=torch.float32).unsqueeze(1)
            x_tensor = torch.stack(x_list[:-1], dim=0)
            G_tensor = torch.tensor(G, dtype=torch.float32).unsqueeze(1)
            v_pred = critic_net(t_tensor, x_tensor)
            critic_loss = torch.sum((v_pred - G_tensor) ** 2)
            critic_loss_sum += critic_loss

        # Update networks after processing batch_size episodes.
        actor_optimizer.zero_grad()
        actor_loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=1.0)
        critic_optimizer.step()

        actor_losses.append(actor_loss_sum.item())
        actor_losses_abs.append(loss_500.item())
        critic_losses.append(critic_loss_sum.item())

        # Print progress: print only for the first epoch and every 5 epochs.
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"[Actor-Critic] Epoch {epoch + 1}: Actor Loss = {loss_500.item():.4e}, Critic Loss = {critic_loss_sum.item():.4e}")

    return actor_losses_abs, critic_losses


# -------------------------
# 6. Main Function: Train Actor-Critic and Perform Testing
# -------------------------
if __name__ == "__main__":
    # Fix random seeds for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)
    
    # System parameters (strictly following the problem requirements): T=0.5, τ=0.5, γ=1, D=I (identity matrix)
    H = np.array([[0.5, 0.5],
                  [0.0, 0.5]], dtype=np.float32)
    M = np.array([[1.0, 1.0],
                  [0.0, 1.0]], dtype=np.float32)
    sigma = 0.5 * np.eye(2, dtype=np.float32)
    C = np.array([[1.0, 0.1],
                  [0.1, 1.0]], dtype=np.float32)
    D = np.eye(2, dtype=np.float32)  # D = I
    R = 10.0 * np.array([[1.0, 0.3],
                         [0.3, 1.0]], dtype=np.float32)
    T = 0.5
    time_grid = np.linspace(0, T, 1001)
    reg_tau = 0.5
    prior_gamma = 1.0
    
    # Initialize Soft LQR solver.
    soft_solver = SoftLQRSolver(H, M, sigma, C, D, R, T, time_grid, reg_tau, prior_gamma)
    soft_solver.solve_Riccati_ODE()
    
    # Initialize Actor and Critic networks and their respective optimizers.
    actor_net = ActorNetwork(state_dim=2, hidden_size=512)
    critic_net = CriticNetwork(input_dim=3, hidden_dim=512)
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-3)
    
    # Training parameters.
    N = 100
    dt = T / N
    
    print("Starting Actor-Critic training ...")
    actor_losses_abs, critic_losses = train_actor_critic_epoch(
        actor_net, critic_net, actor_optimizer, critic_optimizer,
        soft_solver, num_epochs=50, batch_size=500, N=100
    )

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    def plot_actor_vs_optimal(actor_net, soft_solver, D, M, T=0.5):
        x1_vals = np.linspace(-2, 2, 30)
        x2_vals = np.linspace(-2, 2, 30)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        U1_actor = np.zeros_like(X1)
        U2_actor = np.zeros_like(X2)
        U1_opt = np.zeros_like(X1)
        U2_opt = np.zeros_like(X2)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                x = np.array([X1[i, j], X2[i, j]], dtype=np.float32)
                x_tensor = torch.tensor(x).unsqueeze(0)
                t_tensor = torch.tensor([[0.0]], dtype=torch.float32)

                # Actor outputs the mean control.
                with torch.no_grad():
                    mean, _ = actor_net(t_tensor, x_tensor)
                    a_actor = mean.numpy().squeeze()
                U1_actor[i, j], U2_actor[i, j] = a_actor

                # Analytical optimal control.
                S0, _ = soft_solver._get_S_at_time(0.0)
                a_opt = -np.linalg.inv(D) @ M.T @ S0 @ x
                U1_opt[i, j], U2_opt[i, j] = a_opt

        # Plot Actor control u1.
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X1, X2, U1_actor, cmap='viridis')
        ax1.set_title("Actor Control $u_1$ (mean)")
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax1.set_zlabel("$u_1$")

        # Plot analytical optimal control u1.
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X1, X2, U1_opt, cmap='plasma')
        ax2.set_title("Optimal Control $u_1$")
        ax2.set_xlabel("$x_1$")
        ax2.set_ylabel("$x_2$")
        ax2.set_zlabel("$u_1$")

        plt.show()
    plot_actor_vs_optimal(actor_net, soft_solver, D, M, T)

    def plot_critic_vs_true_value(critic_net, soft_solver, T=0.0):
        x1_vals = np.linspace(-2, 2, 30)
        x2_vals = np.linspace(-2, 2, 30)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        V_critic = np.zeros_like(X1)
        V_true = np.zeros_like(X1)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                x = np.array([X1[i, j], X2[i, j]], dtype=np.float32)
                x_tensor = torch.tensor(x).unsqueeze(0)
                t_tensor = torch.tensor([[T]], dtype=torch.float32)

                # Critic network predicted value function.
                with torch.no_grad():
                    v_pred = critic_net(t_tensor, x_tensor)
                V_critic[i, j] = v_pred.item()

                # Analytical true value function.
                x_torch = torch.tensor(x).unsqueeze(0)
                v_true = soft_solver.compute_value_function(t_tensor, x_torch)
                V_true[i, j] = v_true.item()

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X1, X2, V_critic, cmap='viridis')
        ax1.set_title("Critic Value $v(t,x)$ (Predicted)")
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax1.set_zlabel("$v$")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X1, X2, V_true, cmap='plasma')
        ax2.set_title("True Value $v(t,x)$ (Analytical)")
        ax2.set_xlabel("$x_1$")
        ax2.set_ylabel("$x_2$")
        ax2.set_zlabel("$v$")

        plt.show()
    plot_critic_vs_true_value(critic_net, soft_solver, T=0.0)

    # Plot Actor Loss.
    plt.figure(figsize=(8, 4))
    plt.plot(actor_losses_abs, label="Actor Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Actor Loss (absolute)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Critic Loss.
    plt.figure(figsize=(8, 4))
    plt.plot(critic_losses, label="Critic Loss", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
