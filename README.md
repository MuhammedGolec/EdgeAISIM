# EdgeAISim  :shipit:

In this work, we present EdgeAISim, a Python-based toolkit for simulating and modelling AI models in edge computing environments. Edge computing, by bringing processing and storage closer to the network edge, minimises latency and bandwidth usage, meeting the demands of IoT applications. EdgeAISim extends EdgeSimPy and incorporates AI models like Multi-Armed Bandit and Deep Q-Networks to optimise power usage and task migration. The toolkit outperforms baseline worst-fit algorithms, showcasing its potential in sustainable edge computing by significantly reducing power consumption and enhancing task management for various large-scale scenarios.


![Fig2](https://github.com/MuhammedGolec/EdgeAISIM/assets/61287653/41509cd3-cb06-437d-b3ef-c8c0642aa3e4)

Figure 1: EdgeAISim Architecture 

Key Components:
1. Base Stations:
Provide network connectivity to mobile computing devices within their respective coverage areas.
Crucial to ensure equal connectivity within each cell's coverage area.
2. Network Switches:
Establish wired connections between base stations and edge servers.
Model task migrations as network flows, scheduling bandwidth using the Max-Min fairness algorithm.
3. Modelling of Resources:
Edge servers host services and are modeled to consume power based on CPU, RAM, and hard disk utilization.
Incorporates various power consumption models (Linear, Quadratic, Cubic) based on utilization parameters.
4. Users:
Consumers of services hosted on edge servers.
Mobility models define user movement and base station access changes for service consumption.
5. Modelling of Tasks:
Tasks represent applications or services with specific resource requirements (CPU and memory demands).
Allocation of tasks to edge servers impacts power consumption and resource utilization.
6. Modelling of AI Models:
AI module integrates Reinforcement Learning (RL)-based task migration algorithms.
Effective task allocation and migration strategies are discussed in the provided documentation.
EdgeAISim Architecture:
The architecture of EdgeAISim is designed to extend the EdgeSimPy framework and introduces AI-based simulation models for various crucial aspects of edge computing:

Task Scheduling
Energy Management
Service Migration
Network Flow Scheduling
Mobility Support


![Fig4](https://github.com/MuhammedGolec/EdgeAISIM/assets/61287653/796cd491-d0f6-43b0-af0b-0fa825e93199)



Figure 2 provides an insightful visualization of the simulation data flow within EdgeAISim. The process begins as the scheduler receives user requests, meticulously processes them, and intelligently selects the suitable AI module. Subsequently, the designated AI Module takes charge of managing the edge server and effectively schedules the task. Upon completion, task results are transmitted back to the scheduler, which promptly relays them to the respective user.




## Advantages of EdgeAISIM

- EdgeAISim tackles the distinctive challenges associated with simulating AI-driven, energy-efficient resource management policies in edge computing systems.

- EdgeAISim boasts a comprehensive simulation capability, covering a wide range of edge computing functionalities, including task scheduling, service migration, and network flow scheduling.

- EdgeAISim offers robust support for energy management and mobility requirements in the context of edge computing.

- With EdgeAISim, researchers can explore the full spectrum of edge computing functions, making it a versatile tool for in-depth assessments and experiments.

## Explanation for Advanced AI Algorithms

Multi-Arm Bandit Algorithm:

Aims to maximize cumulative rewards by balancing exploration and exploitation in unknown reward scenarios.
Utilizes the Upper Confidence Bound (UCB) algorithm to achieve this balance.

** Deep Q-Networks: Combines Deep Neural Networks (DNN) with Q-Learning to handle large state spaces efficiently.
Enables optimal action-selection strategies in complex, high-dimensional input data environments.
Provides effective task performance improvement through interactions with the environment.

** Graphical Neural Network (GNN): Specifically designed for processing graph-structured data.
Captures dependencies and relationships between entities in graph data.
Enhances node representations for downstream tasks, demonstrating effectiveness in experimental evaluation.

** Actor-Critic Algorithm: Combines value-based and policy-based methods to improve policies and estimate action values.
Utilizes actor and critic networks for policy improvement and value estimation, respectively.
Reinforcement learning framework for effective policy learning in the given task.
Implementation in EdgeAISim Environment

Multi-Arm Bandit for Edge Server Selection: Models each edge server as an arm, choosing servers with minimal power consumption for task migration.
Demonstrates superior performance, choosing servers efficiently with greater frequency.
 
Deep Q-Networks for Server Selection:
Determines the best server for service migration based on server features.
Achieves significant reduction in power consumption, improving service migration decisions.
 Graphical Neural Network for Service Migration:

Models edge servers and links as a graph, leveraging GNN for server selection.
Enhances service migration decisions based on aggregated information from adjacent edge servers.
MAB :
Multi-Armed Bandit for Services: Each service is represented by a Multi-Armed Bandit, with each "arm" symbolizing an edge server.
Task Migration: Tasks can be migrated to these servers by "pulling" the arms, resulting in a received reward inversely proportional to the server's current power consumption.
Tracking Service Migration: The system keeps track of how often each service is migrated to a server over time.
Favoring Efficiency: Servers with lower power consumption while running a service are increasingly selected, promoting energy-efficient choices.

** Q - Learning: Deep Q-Network for Server Selection: A Deep Q-network (DQN) is employed to determine the optimal server for migrating services from the queue.
Input Features: The DQN takes a feature vector as input, comprising available CPU, available RAM, available Disk space, and the current power consumption of the edge server.
Q-Value Output: The DQN processes the concatenated feature vectors from all edge servers and produces a Q-value for each server.
Server Selection and Reward: The server with the highest Q-value is chosen for service migration, and the reward is calculated as the sum of the inverses of the selected server's power consumption.
GNN+DQN;
Graph-Based Modeling: The edge servers and their interconnections are represented as a graph, which is input to a Graph Neural Network (GNN).
Node Feature Vectors: Each edge server corresponds to a graph node with a feature vector encompassing available CPU, available RAM, available Disk space, and current power consumption.
Message Passing: The GNN utilizes message passing to gather information from neighboring edge servers in the graph.
Feature Vector for DQN: The GNN outputs a feature vector that is subsequently fed into the Deep Q-Network (DQN) for decision-making in the system.

** Probabilistic actor-critic: Feature Vector Input: The CPU demand, memory demand, hard disk demand, and power consumption of each edge server are combined into a feature vector.
Actor and Critic Deep Networks: These feature vectors are separately fed into an actor deep network and a critic deep network.
Actor's Role: The actor-network generates a probability distribution over the servers, indicating the likelihood of selecting each server for placing a service.
Critic's Role: The critic network calculates the 'Q-value,' representing the value of the current state based on the probability vector. This value, along with the reward, is used for updating the system's decision-making strategy.

## Implementation and explanation for EdgeAISim codes



Prerequisites:  
* Pip,Python 3.7+  
* Pytorch, Edgesimpy  
  - Use `pip install -q git+https://github.com/EdgeSimPy/EdgeSimPy.git@v1.1.0 ` for installing Edgesimpy
  - Use `pip install pytorch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1`



Steps to run the project:
1. Clone the project.

# Cite this work
This dataset is part of the following publication, please cite when using this dataset:

Nandhakumar, Aadharsh Roshan, et al. "Edgeaisim: A toolkit for simulation and modelling of ai models in edge computing environments." Measurement: Sensors 31 (2024): 100939.
2. Run any script of your choice out of  
   - ```Worst_Fit_Migration_algorithm.py```  
   - ```MAB_migration_algorithm.py```  
   - ```Qlearning_migration.py```  
   - ```GCN_Q_learning.py```  
   - ```Probabilistic_actor_critic_migration.py```  
