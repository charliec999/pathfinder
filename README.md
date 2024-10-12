# Advanced GridWorld DQN

This project implements a Deep Q-Network (DQN) agent for a dynamic GridWorld environment with adjustable grid size, obstacle density, and dynamic obstacles.

## Features

- DQN with LSTM layers
- Prioritized Experience Replay
- Dynamic obstacle generation
- Analysis and visualization tools

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/charliec999/pathfinder.git
cd pathfinder
pip install -r requirements.txt
```

## Usage

Train and evaluate the agent:

```bash
python main.py
```

## Key Components

- `AdvancedGridWorld`: Custom Gym environment
- `AdvancedDQN`: Neural network model with LSTM layers
- `PrioritizedReplayBuffer`: Prioritized experience replay
- `AdvancedDQNAgent`: DQN algorithm implementation

## Analysis Tools

- Training progress visualization
- Q-value and state visitation analysis
- Action distribution, feature importance, and hyperparameter sensitivity

## Future Work

- Multi-agent scenarios
- Integration of PPO, SAC