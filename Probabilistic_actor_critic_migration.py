from edge_sim_py import *
import math
import os
import random
import msgpack
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DDPG import DDPG
import torch
from torch.distributions import Categorical

def custom_collect_method(self) -> dict: # Custom collect method to measure the power consumption of each server
    metrics = {
        "Instance ID": self.id,
        "Power Consumption": self.get_power_consumption(),
    }
    return metrics


global agent
global reward_list
global reward_count_list
global reward_count

reward_list = list()
power_list = list() # List to store total power consumption everytime the task scheduling algorithm is used



def my_algorithm(parameters):
    
    print("\n\n")
    total_reward = 0
    total_power = 0 #We sum the power consumption after migrating each service

    for service in Service.all(): #Iterate over every service
        
    
        if not service.being_provisioned: #If service needs to be migrated

             #Initialise our state vector, which is the concatenation of the cpu,memory,disk utilisation and current power consumption
            state_vector = []

            for edge_server in EdgeServer.all():
                edge_server_cpu = edge_server.cpu
                edge_server_memory = edge_server.memory
                edge_server_disk = edge_server.disk
                power = (edge_server_cpu * edge_server_memory * edge_server_disk) ** (1 / 3)
                vector = [edge_server_cpu, edge_server_memory, edge_server_disk, power]
                state_vector = state_vector + vector


            #Pass the state vector to our actor network, and retrieve the action, as well as the action probabilities
            state_vector = np.array(state_vector)    
            probs, action = agent.select_action(state_vector)

            #print(action)

            if EdgeServer.all()[action[0]] == service.server:  #To conserve resources, we don't want to migrate back to our host
                break


            print(f"[STEP {parameters['current_step']}] Migrating {service} From {service.server} to {EdgeServer.all()[action[0]]}")

            #Migrate service to new edgeserver
            service.provision(target_server=EdgeServer.all()[action[0]])

            

            #Get our next state, after taking action
            next_state_vector = []
            reward = 0
            power = 0



            for edge_server in EdgeServer.all():
                edge_server_cpu = edge_server.cpu
                edge_server_memory = edge_server.memory
                edge_server_disk = edge_server.disk
                power = (edge_server_cpu * edge_server_memory * edge_server_disk) ** (1 / 3)
                vector = [edge_server_cpu, edge_server_memory, edge_server_disk, power]
                next_state_vector = next_state_vector + vector
                reward = reward + (1/edge_server.get_power_consumption()) #Our reward is the inverse of the edge server's power consumption
                power = power + edge_server.get_power_consumption() #get the sum of powerconsumption of each edge server


            next_state_vector = np.array(next_state_vector) #get our next state vector
            agent.replay_buffer.push((state_vector, next_state_vector, probs, action, reward, np.float(0))) #add current episode into replay buffer
#            agent.update(state_vector,action,next_state_vector,reward,False)

            #print(reward)
            total_reward += reward
            total_power += power #Sum our power consumption
    

    reward_list.append(total_reward)
    power_list.append(total_power) #Append power consumption to power list for plotting
#    agent.epsilon*=agent.epsilon_decay
    agent.update()




def stopping_criterion(model: object):    
    # As EdgeSimPy will halt the simulation whenever this function returns True,
    # its output will be a boolean expression that checks if the current time step is 600
    return model.schedule.steps == 1000




simulator = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
)

# Loading a sample dataset
simulator.initialize(input_file="sample_dataset1.json")

EdgeServer.collect = custom_collect_method

#Initialise of DQN agent with state and action dimension
#Here, state is the current cpu, memory and disk utilisation of the server, and action space is the choice of edge server
#i.e. the Edge server with the maximum Q- value will be migrated to
agent = DDPG(len(EdgeServer.all()) * 4, len(EdgeServer.all()))

# Executing the simulation
simulator.run_model()

#Retrieving logs dataframe for plot
logs = pd.DataFrame(simulator.agent_metrics["EdgeServer"])
print(logs)

df = logs

edge_server_ids = df['Instance ID'].unique()

# Determine the number of subplots based on the number of EdgeServers
num_subplots = len(edge_server_ids) + 1  # Add 1 for the rewards subplot

# Create subplots with the desired layout
fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4*num_subplots), sharex=True)

# Iterate over each EdgeServer and plot the data in the corresponding subplot
for i, edge_server_id in enumerate(edge_server_ids):
    # Filter the data for the current EdgeServer
    edge_server_data = df[df['Instance ID'] == edge_server_id]

    # Extract the timestep and power consumption values
    timesteps = edge_server_data['Time Step']
    power_consumption = edge_server_data['Power Consumption']

    # Plot the power consumption data for the current EdgeServer in the corresponding subplot
    axes[i].plot(timesteps, power_consumption, label=f"EdgeServer {edge_server_id}")

    # Set the subplot title and labels
    axes[i].set_title(f"Power Consumption - EdgeServer {edge_server_id}")
    axes[i].set_ylabel("Power Consumption")
    axes[i].legend()

# Create a separate subplot for the rewards
# rewards_subplot = axes[-2]
# reward_count_list = list(range(1, len(reward_list) + 1))
# rewards_subplot.plot(reward_count_list, reward_list)
# rewards_subplot.set_title("Rewards")
# rewards_subplot.set_xlabel("Reward Count")
# rewards_subplot.set_ylabel("Reward")


power_subplot = axes[-1]
power_count_list = list(range(1, len(power_list) + 1))
power_subplot.plot(power_count_list, power_list)
power_subplot.set_title("Power Consumption")
power_subplot.set_xlabel("Power_step Count")
power_subplot.set_ylabel("Power")

# Adjust the spacing between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Display the plot
#plt.show()
plt.savefig('Probabilistic_actor_critic_migration_power_consumption_final_uncropped.png')