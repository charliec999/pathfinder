# Advanced GridWorld DQN

An implementation of a Deep Q-Network (DQN) agent for an advanced GridWorld environment.

## Overview

This project implements an advanced DQN agent capable of navigating a dynamic GridWorld environment. The environment features adjustable grid size, obstacle density, and dynamic obstacles.

## Features

- Advanced DQN with LSTM layers
- Prioritized Experience Replay
- Dynamic obstacle generation
- Comprehensive analysis and visualization tools

## Installation

Clone the repository and install the required packages:

```
git clone https://github.com/yourusername/advanced-gridworld-dqn.git
cd advanced-gridworld-dqn
pip install -r requirements.txt
```

## Usage

Run the main script to train and evaluate the agent:

```
python main.py
```

## Key Components

- `AdvancedGridWorld`: Custom Gym environment
- `AdvancedDQN`: Neural network model with LSTM layers
- `PrioritizedReplayBuffer`: Implementation of prioritized experience replay
- `AdvancedDQNAgent`: Agent class implementing the DQN algorithm

## Analysis Tools

- Training progress visualization
- Q-value and state visitation analysis
- Action distribution and feature importance analysis
- Hyperparameter sensitivity analysis
- Learning dynamics visualization

## Results

[Include key performance metrics and findings]

## Future Work

- Multi-agent scenarios
- Integration of additional RL algorithms (PPO, SAC)
- Enhanced environmental dynamics

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.