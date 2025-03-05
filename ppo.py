import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
from config_sender import configurations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def save_model(agent, filename_policy, filename_value):
    torch.save(agent.policy.state_dict(), filename_policy)
    torch.save(agent.value_function.state_dict(), filename_value)
    # print("Model saved successfully.")


def load_model(agent, filename_policy, filename_value):
    agent.policy.load_state_dict(torch.load(filename_policy, map_location=torch.device('cpu')))
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    agent.value_function.load_state_dict(torch.load(filename_value, map_location=torch.device('cpu')))
    # print("Model loaded successfully.")

exit_signal = 10 ** 10

import copy
class SimulatorState:
    def __init__(
        self,
        sender_buffer_remaining_capacity=0,
        receiver_buffer_remaining_capacity=0,
        read_throughput_change=0,
        write_throughput_change=0,
        network_throughput_change=0,
        read_thread_change=0,
        write_thread_change=0,
        network_thread_change=0,
        read_thread=1,
        write_thread=1,
        network_thread=1,
        rewards_change=0,
        history_length=5  # Store last 5 states
    ):
        # Current state variables
        self.sender_buffer_remaining_capacity = sender_buffer_remaining_capacity
        self.receiver_buffer_remaining_capacity = receiver_buffer_remaining_capacity
        self.read_thread = read_thread
        self.write_thread = write_thread
        self.network_thread = network_thread
        
        # Historical data
        self.history_length = history_length
        self.throughput_history = {
            'read': [0] * (history_length-1) + [read_throughput_change],
            'write': [0] * (history_length-1) + [write_throughput_change],
            'network': [0] * (history_length-1) + [network_throughput_change]
        }
        self.thread_history = {
            'read': [0] * (history_length-1) + [read_thread_change],
            'write': [0] * (history_length-1) + [write_thread_change],
            'network': [0] * (history_length-1) + [network_thread_change]
        }
        self.reward_history = [0] * (history_length-1) + [rewards_change]

    def copy(self):
        return copy.deepcopy(self)

    def update_state(
            self,
            simulator_state = None
    ):
        if simulator_state is not None:
            self.sender_buffer_remaining_capacity = simulator_state.sender_buffer_remaining_capacity
            self.receiver_buffer_remaining_capacity = simulator_state.receiver_buffer_remaining_capacity
            self.read_thread = simulator_state.read_thread
            self.write_thread = simulator_state.write_thread
            self.network_thread = simulator_state.network_thread

            # Update historical data
            self.throughput_history['read'] = self.throughput_history['read'][1:] + [simulator_state.throughput_history['read'][-1]]
            self.throughput_history['write'] = self.throughput_history['write'][1:] + [simulator_state.throughput_history['write'][-1]]
            self.throughput_history['network'] = self.throughput_history['network'][1:] + [simulator_state.throughput_history['network'][-1]]
            self.thread_history['read'] = self.thread_history['read'][1:] + [simulator_state.thread_history['read'][-1]]
            self.thread_history['write'] = self.thread_history['write'][1:] + [simulator_state.thread_history['write'][-1]]
            self.thread_history['network'] = self.thread_history['network'][1:] + [simulator_state.thread_history['network'][-1]]
            self.reward_history = self.reward_history[1:] + [simulator_state.reward_history[-1]]

    def to_array(self):
        # Convert current state and history to a flat array
        current_state = np.array([
            self.sender_buffer_remaining_capacity,
            self.receiver_buffer_remaining_capacity,
            self.read_thread,
            self.write_thread,
            self.network_thread
        ], dtype=np.float32)
        
        # Add historical data
        history = np.concatenate([
            self.throughput_history['read'],
            self.throughput_history['write'],
            self.throughput_history['network'],
            self.thread_history['read'],
            self.thread_history['write'],
            self.thread_history['network'],
            self.reward_history
        ])
        
        return np.concatenate([current_state, history])

class NetworkOptimizationEnv(gym.Env):
    def __init__(self, black_box_function, state, history_length=5):
        super(NetworkOptimizationEnv, self).__init__()
        self.thread_limits = [1, 20]  # Threads can be between 1 and 10

        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        obs_dim = 5 + 7 * history_length

        # print(f"Observation space dimension: {obs_dim}")
        
        # Define an unbounded Box of shape (obs_dim,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.history_length = history_length
        self.get_utility_value = black_box_function

        self.state = state
        self.max_steps = 5
        self.current_step = 0

        # For recording the trajectory
        self.trajectory = []

    def step(self, action):
        deltas_map = [-3, -1, 0, +1, +3]
        
        read_index, net_index, write_index = action
        # Convert those indexes to actual deltas:
        read_delta = deltas_map[read_index]
        net_delta = deltas_map[net_index]
        write_delta = deltas_map[write_index]


        # 2) Compute new thread counts
        new_read = min(max(self.state.read_thread + read_delta, self.thread_limits[0]), configurations['max_cc']['io'])
        new_network = min(max(self.state.network_thread + net_delta, self.thread_limits[0]), configurations['max_cc']['network'])
        new_write = min(max(self.state.write_thread + write_delta, self.thread_limits[0]), configurations['max_cc']['write'])
        new_thread_counts = [new_read, new_network, new_write]

        # Compute utility and update state
        # print(f"New Thread Counts: {new_thread_counts}")
        utility, new_state, grads, bottleneck_idx = self.get_utility_value(new_thread_counts)
        print(f"Utility: {utility}")

        if utility == exit_signal:
            return self.state.to_array(), exit_signal, grads, bottleneck_idx, True, {}


        self.state.update_state(new_state)

        # Penalize actions that hit thread limits
        penalty = 0
        if new_thread_counts[0] == self.thread_limits[0] or new_thread_counts[0] == configurations['max_cc']['io']:
            penalty -= 0.50  # Adjust penalty value as needed
        if new_thread_counts[1] == self.thread_limits[0] or new_thread_counts[1] == configurations['max_cc']['network']:
            penalty -= 0.50
        if new_thread_counts[2] == self.thread_limits[0] or new_thread_counts[2] == configurations['max_cc']['write']:
            penalty -= 0.50

        # Add penalty for large changes
        # change_penalty = -0.1 * np.sum(np.abs(action)) / self.max_delta
        change_penalty = 0
        
        # Adjust reward
        reward = utility + penalty + change_penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record the state
        self.trajectory.append(self.state.copy())

        # Return state as NumPy array
        return self.state.to_array(), reward, grads, bottleneck_idx, done, {}

    # def reset(self, simulator=None):
        
    #     self.current_step = 0
    #     self.trajectory = [self.state.copy()]

    #     return self.state.to_array()
    
    def reset(self, is_inference = False):
        print(f'is_inference: {is_inference}')
        if not is_inference:
            sender_buffer_remaining_capacity = self.state.sender_buffer_remaining_capacity
            receiver_buffer_remaining_capacity = self.state.receiver_buffer_remaining_capacity
            read_thread = np.random.randint(3, 19)
            network_thread = np.random.randint(3, 19)
            write_thread = np.random.randint(3, 19)
            self.state = SimulatorState(
                sender_buffer_remaining_capacity=sender_buffer_remaining_capacity,
                receiver_buffer_remaining_capacity=receiver_buffer_remaining_capacity,
                read_thread=read_thread,
                network_thread=network_thread,
                write_thread=write_thread,
                history_length=self.history_length
            )
        
        self.current_step = 0
        self.trajectory = [self.state.copy()]

        return self.state.to_array()

class ResidualBlock(nn.Module):
    def __init__(self, size, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.activation = activation()

    def forward(self, x):
        # Save the input (for the skip connection)
        residual = x
        
        # Pass through two linear layers with activation
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Add the original input (residual connection)
        out += residual
        
        # Optionally add another activation at the end
        out = self.activation(out)
        return out
    
class PhysicsAwarePolicyDiscrete(nn.Module):
    def __init__(self, state_dim, num_actions=5, num_heads=8, num_layers=3):
        super().__init__()
        # Existing layers
        self.embedding = nn.Linear(state_dim, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action head remains
        self.action_head = nn.Linear(512, 3 * num_actions)
        
        # New components
        self.bottleneck_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Read/Network/Write bottleneck probs
            nn.Softmax(dim=-1)
        )
        
        self.grad_estimator = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # ∂reward/∂(read,net,write)
            nn.Tanh()  # Normalized gradients
        )
        
        self.to(device)

    def forward(self, state):
        x = torch.tanh(self.embedding(state))
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(0)
        
        # Original action logits
        action_logits = self.action_head(x).view(-1, 3, 5).float()
        
        # New outputs
        bottleneck_probs = self.bottleneck_head(x).float()
        gradients = self.grad_estimator(x).float()
        
        return action_logits, bottleneck_probs, gradients
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, num_heads=8):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Linear(state_dim, 512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.to(device)
        
    def forward(self, state):
        x = torch.tanh(self.embedding(state))
        x = x.unsqueeze(0)  # Add sequence dimension
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)
        
        value = self.fc_layers(x)
        return value

# class ResidualBlock(nn.Module):
#     def __init__(self, size, activation=nn.ReLU):
#         super(ResidualBlock, self).__init__()
#         self.fc1 = nn.Linear(size, size)
#         self.fc2 = nn.Linear(size, size)
#         self.activation = activation()

#     def forward(self, x):
#         # Save the input (for the skip connection)
#         residual = x
        
#         # Pass through two linear layers with activation
#         out = self.fc1(x)
#         out = self.activation(out)
#         out = self.fc2(out)
        
#         # Add the original input (residual connection)
#         out += residual
        
#         # Optionally add another activation at the end
#         out = self.activation(out)
#         return out
    
    
# class PhysicsAwarePolicyDiscrete(nn.Module):
#     def __init__(self, state_dim, num_actions=5):
#         super().__init__()
#         # Existing layers
#         self.fc1 = nn.Linear(state_dim, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 256)
        
#         # Action head remains
#         self.action_head = nn.Linear(256, 3 * num_actions)
        
#         # New components
#         self.bottleneck_head = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3),  # Read/Network/Write bottleneck probs
#             nn.Softmax(dim=-1)
#         )
        
#         self.grad_estimator = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3),  # ∂reward/∂(read,net,write)
#             nn.Tanh()  # Normalized gradients
#         )
        
#         self.to(device)

#     def forward(self, state):
#         x = torch.tanh(self.fc1(state))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x))
        
#         # Original action logits
#         action_logits = self.action_head(x).view(-1, 3, 5).float()
        
#         # New outputs
#         bottleneck_probs = self.bottleneck_head(x).float()
#         gradients = self.grad_estimator(x).float()
        
#         return action_logits, bottleneck_probs, gradients
    
# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.value_head = nn.Linear(256, 1)
#         self.to(device)

#     def forward(self, state):
#         """
#         state: shape [batch_size, state_dim]
#         Returns: value estimate of shape [batch_size, 1]
#         """
#         x = torch.tanh(self.fc1(state))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x))
#         value = self.value_head(x)
#         return value

class PPOAgentDiscrete:
    def __init__(self, 
                 state_dim, 
                 lr=1e-3, 
                 gamma=0.99, 
                 eps_clip=0.2,
                 K_epochs=10,              # CHANGE #1: new arg
                 mini_batch_size=64):      # CHANGE #2: new arg
        self.policy = PhysicsAwarePolicyDiscrete(state_dim)
        self.policy_old = PhysicsAwarePolicyDiscrete(state_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function = ValueNetwork(state_dim)
        # Add gradient normalizer
        self.grad_norm = nn.InstanceNorm1d(3)
        
        # Modified optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr},
            {'params': self.grad_norm.parameters(), 'lr': lr*0.1}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()
        self.action_values = [-3, -1, 0, 1, 3]
        
        self.K_epochs = K_epochs                # store them
        self.mini_batch_size = mini_batch_size

    def select_action(self, state, is_inference=False):
        # print(f"Select Action: {state}")
        state = torch.FloatTensor(state).unsqueeze(0).to(device)   # [1, obs_dim]
        logits, bottlenecks, gradients = self.policy_old(state)
            
        # Gradient-informed action modulation
        grad_impact = 0.3 * self.grad_norm(gradients.unsqueeze(0)).squeeze(0)
        modulated_logits = logits + grad_impact.unsqueeze(-1)
        probs = torch.softmax(modulated_logits, dim=-1)

        if is_inference:
            discrete_actions = torch.argmax(probs, dim=-1)  # [1, 3] each in {0..4}
        else:
            dist = torch.distributions.Categorical(probs)
            discrete_actions = dist.sample()                # [1, 3]

        # Convert discrete_actions -> log_probs
        log_probs = torch.log_softmax(logits, dim=-1)       # [1, 3, 5]
        chosen_log_probs = torch.gather(
            log_probs, dim=-1, index=discrete_actions.unsqueeze(-1)
        ).squeeze(-1)                                       # [1, 3]
        chosen_log_probs = chosen_log_probs.sum(dim=1)      # [1]
        logprob_scalar = chosen_log_probs.item()            # float
        bottleneck = torch.argmax(bottlenecks, dim=1).item() # int in {0,1,2}

        # Convert the discrete_actions => environment thread_changes
        # shape [1,3], so take row 0 => shape [3]
        actions_np = discrete_actions[0].cpu().numpy()         # e.g. [2,4,1]
        thread_changes = np.array(
            [self.action_values[a] for a in actions_np],    
            dtype=np.int32
        )  # e.g. if action_values=[-3,-1,0,1,3], then [0,3,1] => [0, +3, -1]

        return thread_changes, logprob_scalar, discrete_actions[0].cpu().numpy(), bottleneck, gradients.detach().cpu().numpy()[0]


    def update(self, memory):
        states = torch.stack(memory.states).to(device)   # shape [N, state_dim]
        actions = torch.tensor(memory.actions, dtype=torch.long).to(device)  # shape [N, 3]
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device) # shape [N]
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device) # shape [N]

        # ---- 1) Compute discounted returns ----
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Optionally normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Pre-compute state-values
        with torch.no_grad():
            state_values = self.value_function(states).squeeze()  # shape [N]

        # Advantage
        advantages = returns - state_values

        # ---- 2) Multiple epochs over the batch ----
        full_batch_size = len(states)
        indices = np.arange(full_batch_size)

        for _ in range(self.K_epochs):               # Repeat K_epochs
            np.random.shuffle(indices)

            for start in range(0, full_batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                # Extract mini-batch
                mb_states      = states[mb_indices]
                mb_actions     = actions[mb_indices]       # shape [MB, 3]
                mb_old_logprob = old_logprobs[mb_indices]  # shape [MB]
                mb_returns     = returns[mb_indices]       # shape [MB]
                mb_advantages  = advantages[mb_indices]    # shape [MB]

                # ---- Forward pass for new log-probs ----
                logits, bottlenecks, gradients = self.policy(mb_states)                    # shape [MB, 3, 5]
                new_logprobs_all = torch.log_softmax(logits, dim=-1)  # shape [MB, 3, 5]
                # Gather log-probs of chosen actions
                selected_logprobs = new_logprobs_all.gather(
                    2, mb_actions.unsqueeze(2)
                ).squeeze(-1)  # shape [MB, 3]

                # Sum across the 3 dimensions (read/network/write)
                new_logprobs = selected_logprobs.sum(dim=1)  # shape [MB]

                # Entropy
                probs_all = torch.softmax(logits, dim=-1)   # shape [MB, 3, 5]
                entropy_all = -(probs_all * new_logprobs_all).sum(dim=-1) # shape [MB, 3]
                entropy = entropy_all.sum(dim=1).mean()      # mean across mini-batch

                # Value
                V = self.value_function(mb_states).squeeze()  # shape [MB]

                # Surrogate ratio
                ratios = torch.exp(new_logprobs - mb_old_logprob)  # shape [MB]

                # PPO objectives
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                # Critic loss
                value_loss = self.MseLoss(V, mb_returns)

                # Actor loss: negative of clipped surrogate, plus value loss, minus entropy bonus
                actor_loss  = -torch.min(surr1, surr2).mean()

                subset_bottleneck_true = [memory.bottleneck_true[i] for i in mb_indices]
                subset_grad_true       = [memory.grad_true[i]       for i in mb_indices]

                bottleneck_loss = torch.nn.functional.cross_entropy(
                    bottlenecks,  # shape [MB, 3]
                    torch.tensor(subset_bottleneck_true).to(device)  # shape [MB]
                )

                grad_loss = torch.nn.functional.mse_loss(
                    gradients,     # shape [MB, 3]
                    torch.tensor(subset_grad_true).to(device)  # shape [MB, 3]
                )
                
                # Combined loss
                total_loss = (
                    actor_loss + 
                    0.5 * value_loss +
                    0.2 * bottleneck_loss +
                    0.3 * grad_loss -
                    0.01 * entropy
                )
                
                # Backprop through all components
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # ---- 3) After the multiple epochs, update old policy ----
        self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.bottleneck_true = []
        self.grad_true = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.bottleneck_true[:]
        del self.grad_true[:]


from tqdm import tqdm

def train_ppo(env, agent, max_episodes=1000, is_inference=False):
    memory = Memory()
    total_rewards = []
    for episode in tqdm(range(1, max_episodes + 1), desc="Episodes"):
        # print(f"Episode {episode}")
        state = env.reset(is_inference)
        episode_reward = 0
        exit_flag = False
        for t in range(env.max_steps):
            print(f"Step {t}")
            thread_changes, logprob_scalar, action_indices, _, _ = agent.select_action(state)
            # print(f"Thread Changes: {thread_changes}")
        
            # Step environment with thread_changes
            next_state, reward, grads, bottleneck_idx, done, _ = env.step(thread_changes)
            print(f"Reward: {reward}")

            if reward == exit_signal:
                exit_flag = True
                break

            memory.states.append(torch.FloatTensor(state).to(device))
            memory.actions.append(action_indices)       # This is crucial! action_indices in [0..4]
            memory.logprobs.append(logprob_scalar)
            memory.rewards.append(reward)
            memory.bottleneck_true.append(bottleneck_idx)
            memory.grad_true.append(grads)

            state = next_state
            if t==0:
                episode_reward += reward
            if done:
                break

        if not done:
            agent.update(memory)

        # print(f"Episode {episode}\tLast State: {state}\tReward: {reward}")
        with open('episode_rewards_training_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv', 'a') as f:
                f.write(f"Episode {episode}, Last State: {np.round(state[-3:])}, Reward: {reward}\n")

        memory.clear()
        if exit_flag:
            break
        total_rewards.append(episode_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
            save_model(agent, "models/finetune_v" + configurations['model_version'] +"_policy_"+ str(episode) +".pth", "models/finetune_v" + configurations['model_version'] +"_value_"+ str(episode) +".pth")
            print("Model saved successfully.")
    return total_rewards

def plot_rewards(rewards, title, pdf_file):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.xlim(0, len(rewards))
    plt.ylim(-1, 1)
    plt.title(title)
    plt.grid(True)
    
    plt.savefig(pdf_file)  
    plt.close()

import csv

import pandas as pd

def plot_threads_csv(threads_file='threads_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv', optimals = None, output_file='threads_plot.png'):
    optimal_read, optimal_network, optimal_write, _ = optimals
    data = []

    # Read data from threads_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv
    with open(threads_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            data.append([float(value) for value in row[:3]])

    df = pd.DataFrame(data, columns=['Read Threads', 'Network Threads', 'Write Threads'])

    # Compute rolling averages
    rolling_read = df['Read Threads'].rolling(window=15).mean()
    rolling_network = df['Network Threads'].rolling(window=15).mean()
    rolling_write = df['Write Threads'].rolling(window=15).mean()

    # Create subplots for each type
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(rolling_read, label='Read Threads (5-point MA)')
    plt.title('Read Threads (Stable: '+ str(optimal_read) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Thread Count')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(rolling_network, label='Network Threads (5-point MA)', color='orange')
    plt.title('Network Threads (Stable: '+ str(optimal_network) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Thread Count')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(rolling_write, label='Write Threads (5-point MA)', color='green')
    plt.title('Write Threads (Stable: '+ str(optimal_write) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Thread Count')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    # print(f"Saved thread count plot to {output_file}")

    # save average thread count to a file
    with open('average_threads_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv', 'a') as f:
        f.write(f"optimal: {optimal_read}; Actual: {np.mean(df['Read Threads'])}\n")
        f.write(f"optimal: {optimal_network}; Actual: {np.mean(df['Network Threads'])}\n")
        f.write(f"optimal: {optimal_write}; Actual: {np.mean(df['Write Threads'])}\n")

# Function to plot throughputs with rolling averages
def plot_throughputs_csv(throughputs_file='throughputs_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv', optimals = None, output_file='throughputs_plot.png'):
    optimal_throughput = optimals[-1]
    data = []

    # Read data from throughputs_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv
    with open(throughputs_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            data.append([float(value) for value in row[:3]])

    df = pd.DataFrame(data, columns=['Read Throughput', 'Network Throughput', 'Write Throughput'])

    # Compute rolling averages
    rolling_read = df['Read Throughput'].rolling(window=15).mean()
    rolling_network = df['Network Throughput'].rolling(window=15).mean()
    rolling_write = df['Write Throughput'].rolling(window=15).mean()

    # Create subplots for each type
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(rolling_read, label='Read Throughput')
    plt.title('Read Throughput (Stable: '+ str(optimal_throughput) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(rolling_network, label='Network Throughput', color='orange')
    plt.title('Network Throughput (Stable: '+ str(optimal_throughput) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(rolling_write, label='Write Throughput', color='green')
    plt.title('Write Throughput (Stable: '+ str(optimal_throughput) +')')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    # print(f"Saved throughput plot to {output_file}")

    # save average throughput to a file
    with open('average_throughput_dicrete_w_history_minibatch_mlp_deepseek_v' + configurations['model_version'] +'.csv', 'a') as f:
        f.write(f"{np.mean(df['Read Throughput'])}\n")
        f.write(f"{np.mean(df['Network Throughput'])}\n")
        f.write(f"{np.mean(df['Write Throughput'])}\n")

