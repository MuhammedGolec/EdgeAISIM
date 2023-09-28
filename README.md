# EdgeAISIM

In this work, we present EdgeAISim, a Python-based toolkit for simulating and modelling AI models in edge computing environments. Edge computing, by bringing processing and storage closer to the network edge, minimises latency and bandwidth usage, meeting the demands of IoT applications. EdgeAISim extends EdgeSimPy and incorporates AI models like Multi-Armed Bandit and Deep Q-Networks to optimise power usage and task migration. The toolkit outperforms baseline worst-fit algorithms, showcasing its potential in sustainable edge computing by significantly reducing power consumption and enhancing task management for various large-scale scenarios.


![Fig2](https://github.com/MuhammedGolec/EdgeAISIM/assets/61287653/41509cd3-cb06-437d-b3ef-c8c0642aa3e4)





## Advantages of EdgeAISIM

- EdgeAISim tackles the distinctive challenges associated with simulating AI-driven, energy-efficient resource management policies in edge computing systems.

- EdgeAISim boasts a comprehensive simulation capability, covering a wide range of edge computing functionalities, including task scheduling, service migration, and network flow scheduling.

- EdgeAISim offers robust support for energy management and mobility requirements in the context of edge computing.

- With EdgeAISim, researchers can explore the full spectrum of edge computing functions, making it a versatile tool for in-depth assessments and experiments.








## Implementation and explanation for EdgeAISim codes



Prerequisites:  
* Pip,Python 3.7+  
* Pytorch, Edgesimpy  
  - Use `pip install -q git+https://github.com/EdgeSimPy/EdgeSimPy.git@v1.1.0 ` for installing Edgesimpy
  - Use `pip install pytorch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1`



Steps to run the project:
1. Clone the project.
2. Run any script of your choice out of  
   - ```Worst_Fit_Migration_algorithm.py```  
   - ```MAB_migration_algorithm.py```  
   - ```Qlearning_migration.py```  
   - ```GCN_Q_learning.py```  
   - ```Probabilistic_actor_critic_migration.py```  
