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
    def __init__(self, sender_buffer_remaining_capacity=0, receiver_buffer_remaining_capacity=0,
                 read_throughput=0, write_throughput=0, network_throughput=0,
                 read_thread=0, write_thread=0, network_thread=0) -> None:
        self.sender_buffer_remaining_capacity = sender_buffer_remaining_capacity
        self.receiver_buffer_remaining_capacity = receiver_buffer_remaining_capacity
        self.read_throughput = read_throughput
        self.write_throughput = write_throughput
        self.network_throughput = network_throughput
        self.read_thread = read_thread
        self.write_thread = write_thread
        self.network_thread = network_thread

    def copy(self):
        # Return a new SimulatorState instance with the same attribute values
        return SimulatorState(
            sender_buffer_remaining_capacity=self.sender_buffer_remaining_capacity,
            receiver_buffer_remaining_capacity=self.receiver_buffer_remaining_capacity,
            read_throughput=self.read_throughput,
            write_throughput=self.write_throughput,
            network_throughput=self.network_throughput,
            read_thread=self.read_thread,
            write_thread=self.write_thread,
            network_thread=self.network_thread
        )

    def to_array(self):
        # Convert the state to a NumPy array
        return np.array([
            self.sender_buffer_remaining_capacity,
            self.receiver_buffer_remaining_capacity,
            self.read_throughput,
            self.write_throughput,
            self.network_throughput,
            self.read_thread,
            self.write_thread,
            self.network_thread
        ], dtype=np.float32)

class NetworkOptimizationEnv(gym.Env):
    def __init__(self, black_box_function, state, history_length=5):
        super(NetworkOptimizationEnv, self).__init__()
        self.thread_limits = [1, 20]  # Threads can be between 1 and 10

        # Continuous action space: adjustments between -5.0 and +5.0
        self.action_space = spaces.Box(low=np.array([self.thread_limits[0]] * 3),
                               high=np.array([self.thread_limits[1]] * 3),
                               dtype=np.float32)
        
        oneGB = 1024

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, self.thread_limits[0], self.thread_limits[0], self.thread_limits[0]]),
            high=np.array([
                10 * oneGB,
                10 * oneGB,
                np.inf,  # Or maximum possible throughput values
                np.inf,
                np.inf,
                self.thread_limits[1],
                self.thread_limits[1],
                self.thread_limits[1]
            ]),
            dtype=np.float32
        )

        self.history_length = history_length
        self.get_utility_value = black_box_function

        self.state = state
        self.max_steps = 10
        self.current_step = 0

        # For recording the trajectory
        self.trajectory = []

    def step(self, action):
        new_thread_counts = np.clip(np.round(action), self.thread_limits[0], self.thread_limits[1]).astype(np.int32)
        print(new_thread_counts)
        
        # to get random values 
        # read_thread = np.random.randint(3, 19)
        # network_thread = np.random.randint(3, 19)
        # write_thread = np.random.randint(3, 19)
        # new_thread_counts = [read_thread, network_thread, write_thread]
        
        # Compute utility and update state
        utility, self.state = self.get_utility_value(new_thread_counts)

        print(f"Utility: {utility}")

        if utility == exit_signal:
            return self.state.to_array(), exit_signal, True, {}

        # Penalize actions that hit thread limits
        penalty = 0
        if new_thread_counts[0] == self.thread_limits[0] or new_thread_counts[0] == self.thread_limits[1]:
            penalty -= 100  # Adjust penalty value as needed
        if new_thread_counts[1] == self.thread_limits[0] or new_thread_counts[1] == self.thread_limits[1]:
            penalty -= 100
        if new_thread_counts[2] == self.thread_limits[0] or new_thread_counts[2] == self.thread_limits[1]:
            penalty -= 100

        # Adjust reward
        reward = utility + penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record the state
        self.trajectory.append(self.state.copy())

        # Return state as NumPy array
        return self.state.to_array(), reward, done, {}
    
    def reset(self, is_inference = False):
        print(is_inference)
        if not is_inference:
            read_thread = np.random.randint(3, 19)
            network_thread = np.random.randint(3, 19)
            write_thread = np.random.randint(3, 19)
            sender_buffer_remaining_capacity = self.state.sender_buffer_remaining_capacity
            receiver_buffer_remaining_capacity = self.state.receiver_buffer_remaining_capacity
            print(f"READ: {read_thread}; NETWORK: {network_thread}; WRITE: {write_thread}")

            self.state = SimulatorState(
                sender_buffer_remaining_capacity=sender_buffer_remaining_capacity,
                receiver_buffer_remaining_capacity=receiver_buffer_remaining_capacity,
                read_thread=read_thread,
                network_thread=network_thread,
                write_thread=write_thread,
            )
        
        self.current_step = 0
        self.trajectory = [self.state.copy()]

        # Return initial state as NumPy array
        return self.state.to_array()

class PolicyNetworkContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4, num_layers=2):
        super(PolicyNetworkContinuous, self).__init__()
        self.embedding = nn.Linear(state_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.to(device)
        
    def forward(self, state):
        x = torch.tanh(self.embedding(state))
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(0)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, num_heads=4):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Linear(state_dim, 256)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
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

class PPOAgentContinuous:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetworkContinuous(state_dim, action_dim)
        self.policy_old = PolicyNetworkContinuous(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function = ValueNetwork(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, std = self.policy_old(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()

    def update(self, memory):
        states = torch.stack(memory.states).to(device)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device)

        # Compute discounted rewards
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Get new action probabilities
        mean, std = self.policy(states)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        logprobs = logprobs.sum(dim=1)
        old_logprobs = old_logprobs.sum(dim=1)
        entropy = entropy.sum(dim=1)

        ratios = torch.exp(logprobs - old_logprobs)
        state_values = self.value_function(states).squeeze()

        # Compute advantage
        advantages = returns - state_values.detach()

        # Surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.1 * entropy

        # Update policy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]


from tqdm import tqdm

def train_ppo(env, agent, max_episodes=1000, is_inference=False):
    memory = Memory()
    total_rewards = []
    for episode in tqdm(range(1, max_episodes + 1), desc="Episodes"):
        state = env.reset(is_inference)
        episode_reward = 0
        exit_flag = False
        for t in range(env.max_steps):
            action, action_logprob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            print(f"Reward: {reward}")

            if reward == exit_signal:
                exit_flag = True
                break

            memory.states.append(torch.FloatTensor(state).to(device))
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)

            state = next_state
            if t==0:
                episode_reward += reward
            if done:
                break

        if not done:
            agent.update(memory)

        # print(f"Episode {episode}\tLast State: {state}\tReward: {reward}")
        with open('episode_rewards_residual_cl_finetune.csv', 'a') as f:
                f.write(f"Episode {episode}, Last State: {np.round(state)}, Reward: {reward}\n")

        memory.clear()
        if exit_flag:
            break
        total_rewards.append(episode_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
        if episode % 10 == 0:
            save_model(agent, "models/residual_cl_finetune_policy_"+ str(episode) +".pth", "models/residual_cl_finetune_value_"+ str(episode) +".pth")
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

