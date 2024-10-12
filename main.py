import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import scipy.stats as stats

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class AdvancedGridWorld(gym.Env):
    def __init__(self, grid_size=15, obstacle_density=0.3, dynamic_obstacles=True):
        super(AdvancedGridWorld, self).__init__()
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.dynamic_obstacles = dynamic_obstacles
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(4,), dtype=np.float32)
        self.reset()
        
    def reset(self):
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.goal_pos = np.array([self.grid_size-1, self.grid_size-1], dtype=np.float32)
        self.obstacles = self._generate_obstacles()
        self.time_step = 0
        return self._get_state()
    
    def _generate_obstacles(self):
        obstacles = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < self.obstacle_density:
                    if (i, j) != (0, 0) and (i, j) != (self.grid_size-1, self.grid_size-1):
                        obstacles.add((i, j))
        return obstacles
    
    def step(self, action):
        self.time_step += 1
        old_pos = self.agent_pos.copy()
        move = np.array([(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)][action])
        self.agent_pos += move
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
        
        done = False
        if tuple(self.agent_pos.astype(int)) in self.obstacles:
            self.agent_pos = old_pos
            reward = -1
        elif np.array_equal(self.agent_pos, self.goal_pos):
            reward = 100 / (self.time_step ** 0.5)
            done = True
        else:
            reward = -0.1 - 0.01 * np.linalg.norm(self.agent_pos - self.goal_pos)
        
        if self.dynamic_obstacles and random.random() < 0.05:
            self._update_dynamic_obstacles()
        
        info = {'time_step': self.time_step}
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        return np.concatenate([self.agent_pos, self.goal_pos - self.agent_pos])
    
    def _update_dynamic_obstacles(self):
        for obstacle in list(self.obstacles):
            if random.random() < 0.1:
                self.obstacles.remove(obstacle)
                new_pos = (
                    (obstacle[0] + random.randint(-1, 1)) % self.grid_size,
                    (obstacle[1] + random.randint(-1, 1)) % self.grid_size
                )
                if new_pos != (0, 0) and new_pos != (self.grid_size-1, self.grid_size-1):
                    self.obstacles.add(new_pos)

class AdvancedDQN(keras.Model):
    def __init__(self, num_actions, num_hidden_layers=5):
        super(AdvancedDQN, self).__init__()
        self.num_actions = num_actions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = [layers.Dense(128, activation='elu') for _ in range(num_hidden_layers)]
        self.lstm = layers.LSTM(64)
        self.advantage_stream = layers.Dense(num_actions)
        self.value_stream = layers.Dense(1)
        self.dropout = layers.Dropout(0.2)
        
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.dropout(x)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm(x)
        advantage = self.advantage_stream(x)
        value = self.value_stream(x)
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
    
    def push(self, *args):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class AdvancedDQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=3e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        self.model = AdvancedDQN(num_actions)
        self.target_model = AdvancedDQN(num_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        self.update_target_network()
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state = tf.expand_dims(state, 0)
        q_values = self.model(state)
        return tf.argmax(q_values[0]).numpy()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    @tf.function
    def update_model(self, states, actions, rewards, next_states, dones, importance_weights):
        next_q_values = self.target_model(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + (1 - tf.cast(dones, tf.float32)) * self.gamma * max_next_q_values
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            predicted_q_values = tf.reduce_sum(q_values * action_masks, axis=1)
            losses = tf.square(target_q_values - predicted_q_values)
            weighted_losses = importance_weights * losses
            loss = tf.reduce_mean(weighted_losses)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss, tf.abs(target_q_values - predicted_q_values)
    
    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        transitions, indices, importance_weights = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        states = tf.convert_to_tensor(np.array(batch.state))
        actions = tf.convert_to_tensor(np.array(batch.action))
        rewards = tf.convert_to_tensor(np.array(batch.reward, dtype=np.float32))
        next_states = tf.convert_to_tensor(np.array(batch.next_state))
        dones = tf.convert_to_tensor(np.array(batch.done, dtype=np.float32))
        importance_weights = tf.convert_to_tensor(importance_weights, dtype=tf.float32)
        
        loss, td_errors = self.update_model(states, actions, rewards, next_states, dones, importance_weights)
        self.replay_buffer.update_priorities(indices, td_errors.numpy())
        
        return loss.numpy()
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

def visualize_grid(env, agent, step):
    grid = np.zeros((env.grid_size, env.grid_size, 3))
    for obstacle in env.obstacles:
        grid[obstacle[0], obstacle[1]] = [0.5, 0.5, 0.5]
    grid[int(env.goal_pos[0]), int(env.goal_pos[1])] = [0, 1, 0]
    grid[int(env.agent_pos[0]), int(env.agent_pos[1])] = [1, 0, 0]
    
    plt.clf()
    plt.imshow(grid)
    plt.title(f"Step: {step}, Epsilon: {agent.epsilon:.2f}")
    plt.axis('off')
    plt.pause(0.01)

def train_agent(env, agent, episodes=1000, batch_size=64, update_freq=10, visualize=False):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        losses = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            if len(agent.replay_buffer) > batch_size:
                loss = agent.train(batch_size)
                losses.append(loss)
            
            if step % update_freq == 0:
                agent.update_target_network()
            
            if visualize and episode % 10 == 0:
                visualize_grid(env, agent, step)
        
        agent.update_epsilon()
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_loss = np.mean(losses) if losses else 0
            print(f"Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards_history

def evaluate_agent(env, agent, num_episodes=50):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)

def visualize_q_values(env, agent):
    q_values = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            state = np.array([i, j, env.goal_pos[0] - i, env.goal_pos[1] - j])
            q_values[i, j] = agent.model(tf.expand_dims(state, 0))[0].numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(q_values[:, :, i], cmap='viridis')
        ax.set_title(f'Q-values for action: {actions[i]}')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def plot_training_progress(rewards_history):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def analyze_state_visitation(env, agent, num_episodes=100):
    visitation_counts = np.zeros((env.grid_size, env.grid_size))
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            visitation_counts[int(state[0]), int(state[1])] += 1
            action = agent.get_action(state)
            state, _, done, _ = env.step(action)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(visitation_counts, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.title('State Visitation Heatmap')
    plt.show()

def analyze_action_distribution(agent, num_samples=10000):
    states = np.random.rand(num_samples, 4)
    actions = np.array([agent.get_action(state) for state in states])
    action_counts = np.bincount(actions, minlength=agent.num_actions)
    action_probs = action_counts / num_samples
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(agent.num_actions), action_probs)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.show()

def visualize_neural_network(model):
    dot = tf.keras.utils.model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir="TB")
    dot.get_nodes()[0].set_fillcolor("#E0E0E0")
    dot.get_nodes()[-1].set_fillcolor("#E0E0E0")
    plt.figure(figsize=(12, 12))
    plt.imshow(dot.create_png(), aspect='equal')
    plt.axis('off')
    plt.show()

def perform_sensitivity_analysis(env, state_shape, num_actions, param_ranges, num_trials=3):
    results = {}
    for param, values in param_ranges.items():
        param_results = []
        for value in values:
            trial_rewards = []
            for _ in range(num_trials):
                if param == 'learning_rate':
                    agent = AdvancedDQNAgent(state_shape, num_actions, learning_rate=value)
                elif param == 'gamma':
                    agent = AdvancedDQNAgent(state_shape, num_actions, gamma=value)
                elif param == 'epsilon_decay':
                    agent = AdvancedDQNAgent(state_shape, num_actions, epsilon_decay=value)
                
                rewards = train_agent(env, agent, episodes=100, visualize=False)
                trial_rewards.append(np.mean(rewards[-10:]))
            param_results.append(np.mean(trial_rewards))
        results[param] = param_results
    
    fig, axes = plt.subplots(1, len(param_ranges), figsize=(20, 5))
    for i, (param, values) in enumerate(param_ranges.items()):
        axes[i].plot(values, results[param])
        axes[i].set_title(f'Sensitivity to {param}')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Average Reward')
    plt.tight_layout()
    plt.show()

def analyze_learning_dynamics(env, agent, episodes=500, batch_size=64):
    q_value_changes = []
    td_errors = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_q_changes = []
        episode_td_errors = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            if len(agent.replay_buffer) > batch_size:
                old_q = agent.model(tf.expand_dims(state, 0))[0, action].numpy()
                loss, td_error = agent.train(batch_size)
                new_q = agent.model(tf.expand_dims(state, 0))[0, action].numpy()
                
                episode_q_changes.append(np.abs(new_q - old_q))
                episode_td_errors.append(td_error.numpy().mean())
            
            state = next_state
        
        q_value_changes.append(np.mean(episode_q_changes) if episode_q_changes else 0)
        td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)
        
        if episode % 50 == 0:
            print(f"Episode {episode} completed")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(q_value_changes)
    ax1.set_title('Average Q-value Change per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Q-value Change')
    
    ax2.plot(td_errors)
    ax2.set_title('Average TD Error per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('TD Error')
    
    plt.tight_layout()
    plt.show()

def visualize_value_function(env, agent):
    value_map = np.zeros((env.grid_size, env.grid_size))
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            state = np.array([i, j, env.goal_pos[0] - i, env.goal_pos[1] - j])
            value_map[i, j] = np.max(agent.model(tf.expand_dims(state, 0))[0].numpy())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(value_map, cmap='viridis')
    plt.colorbar(label='Estimated Value')
    plt.title('Value Function Visualization')
    plt.show()

def analyze_feature_importance(agent, num_samples=10000):
    states = np.random.rand(num_samples, 4)
    q_values = agent.model(states).numpy()
    
    feature_importance = np.zeros(4)
    for i in range(4):
        perturbed_states = states.copy()
        perturbed_states[:, i] = np.random.rand(num_samples)
        perturbed_q_values = agent.model(perturbed_states).numpy()
        feature_importance[i] = np.mean(np.abs(q_values - perturbed_q_values))
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Agent X', 'Agent Y', 'Goal X Diff', 'Goal Y Diff'], feature_importance)
    plt.title('Feature Importance Analysis')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.show()

def main():
    env = AdvancedGridWorld(grid_size=15, obstacle_density=0.3, dynamic_obstacles=True)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    agent = AdvancedDQNAgent(state_shape, num_actions)
    
    rewards_history = train_agent(env, agent, episodes=1000, batch_size=64, update_freq=10, visualize=True)
    
    plot_training_progress(rewards_history)
    
    mean_reward, std_reward = evaluate_agent(env, agent)
    print(f"Evaluation Results - Mean Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}")
    
    visualize_q_values(env, agent)
    analyze_state_visitation(env, agent)
    analyze_action_distribution(agent)
    visualize_neural_network(agent.model)
    
    param_ranges = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'gamma': [0.95, 0.99, 0.999],
        'epsilon_decay': [0.995, 0.9975, 0.999]
    }
    perform_sensitivity_analysis(env, state_shape, num_actions, param_ranges)
    
    analyze_learning_dynamics(env, agent)
    visualize_value_function(env, agent)
    analyze_feature_importance(agent)
    
    agent.model.save('advanced_dqn_model.h5')
    print("Model saved successfully.")

def perform_sensitivity_analysis(env, state_shape, num_actions, param_ranges, num_trials=3):
    results = {}
    for param, values in param_ranges.items():
        param_results = []
        for value in values:
            trial_rewards = []
            for _ in range(num_trials):
                if param == 'learning_rate':
                    agent = AdvancedDQNAgent(state_shape, num_actions, learning_rate=value)
                elif param == 'gamma':
                    agent = AdvancedDQNAgent(state_shape, num_actions, gamma=value)
                elif param == 'epsilon_decay':
                    agent = AdvancedDQNAgent(state_shape, num_actions, epsilon_decay=value)
                
                rewards_history = train_agent(env, agent, episodes=100, batch_size=64, update_freq=10)
                avg_reward = np.mean(rewards_history[-10:])
                trial_rewards.append(avg_reward)
            
            avg_trial_reward = np.mean(trial_rewards)
            param_results.append(avg_trial_reward)
        
        results[param] = param_results
    
    return results


if __name__ == "__main__":
    main()