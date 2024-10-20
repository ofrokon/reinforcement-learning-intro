# Reinforcement Learning: Teaching Machines Through Trial and Error

This repository contains Python scripts demonstrating the implementation and visualization of Reinforcement Learning algorithms using PyTorch. It accompanies the Medium post "Reinforcement Learning: Teaching Machines Through Trial and Error".

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Visualizations](#visualizations)
4. [Topics Covered](#topics-covered)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run these scripts, you need Python 3.6 or later. Follow these steps to set up your environment:

1. Clone this repository:
   ```
   git clone https://github.com/ofrokon/reinforcement-learning-intro.git
   cd reinforcement-learning-intro
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate all visualizations and train the RL agents, run:

```
python reinforcement_learning_visualizations.py
```

This will create PNG files for visualizations and print training progress in the console.

## Visualizations

This script generates the following visualizations:

1. `rl_process.png`: Diagram of the Reinforcement Learning process
2. `q_learning_rewards.png`: Learning curve for Q-Learning agent
3. `learned_policy.png`: Visualization of the learned policy in GridWorld

## Topics Covered

1. Reinforcement Learning Basics
2. Q-Learning Algorithm
3. Implementing Q-Learning for a GridWorld environment
4. Visualizing the learned policy
5. Introduction to Deep Q-Learning (DQN)

Each topic is explained in detail in the accompanying Medium post, including Python implementation and visualizations.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your suggested changes. If you're planning to make significant changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For a detailed explanation of Reinforcement Learning concepts and their implementation using PyTorch, check out the accompanying Medium post: [Reinforcement Learning: Teaching Machines Through Trial and Error](https://medium.com/yourusername/reinforcement-learning-teaching-machines-through-trial-and-error)

For questions or feedback, please open an issue in this repository.
