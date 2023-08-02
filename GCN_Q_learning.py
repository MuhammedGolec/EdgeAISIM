from edge_sim_py import *
from GNN import MyGNN
from torch_geometric.data import Data
import torch
from CNNDQN import DQNAgent
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

global network_links_adj_list #Our graph of adjacent network switches
global network_switch_dict #dict of network switches to edge servers
global edge_server_dict #dict of edge servers to network switches
global matrix_list
global network_switch_index #network switch index mapping
global agent
global optimizer
global model

criterion = nn.MSELoss() #Mean Squared error criterion provided by pytorch
reward_list = list()
power_list = list() # List to store total power consumption everytime the task scheduling algorithm is used

def custom_collect_method(self) -> dict: # Custom collect method to measure the power consumption of each server
    metrics = {
        "Instance ID": self.id,
        "Power Consumption": self.get_power_consumption(),
    }
    return metrics




def my_algorithm(parameters):
  
    print("\n\n")
    total_reward = 0
    total_power = 0 #We sum the power consumption after migrating each service
    for service in Service.all(): #Iterate over every service
        

        
        #Create a list of our EdgeServer states
        state_vector = list()
        next_state_vector = list()
        
        
        #Initialise them with zeros
        for _ in range(len(NetworkSwitch.all())):
            state_vector.append(np.zeros(4))
            next_state_vector.append(np.zeros(4))

        state_vector = np.array(state_vector)
        next_state_vector = np.array(next_state_vector)

        if not service.being_provisioned:
            

            
            #We treat each edge server connected to the same network sitch as one node, and hence, sum their current utilizations
            for edge_server in EdgeServer.all():
                edge_server_cpu = edge_server.cpu
                edge_server_memory = edge_server.memory
                edge_server_disk = edge_server.disk
                power = (edge_server_cpu * edge_server_memory * edge_server_disk) ** (1 / 3)
                vector = [edge_server_cpu, edge_server_memory, edge_server_disk, power]
                vector = np.array(vector)
                state_vector[network_switch_index[edge_server_dict[edge_server]]] = np.add(state_vector[network_switch_index[edge_server_dict[edge_server]]],vector)

            
            
            #print(state_vector)

            matrix_array = np.array(matrix_list)

            #print(matrix_array)

            #Create state and adjacency list tensors

            state_vector_tensor  = torch.tensor(state_vector,dtype=torch.float64)
            adjacency_list_tensor = torch.tensor(matrix_array,dtype=torch.int64)

            state_vector_tensor = state_vector_tensor.to(model.lin1.weight.dtype)

            
            #create our dataset and pass to model
            data = Data(x=state_vector_tensor, edge_index=adjacency_list_tensor)
            output = model(data.x, data.edge_index)

#            output = output.unsqueeze(0).unsqueeze(0)


            #pass graph network output to Q network
            output = output.detach()
            action = agent.choose_action(output)

            #To conserve resources, we don't want to migrate back to our host 
            if(service.server == EdgeServer.all()[action]):
                 break

            print(f"[STEP {parameters['current_step']}] Migrating {service} From {service.server} to {EdgeServer.all()[action]}")

            #Migrate service to new edgeserver
            service.provision(target_server=EdgeServer.all()[action])

            reward = 0
            power = 0

            #Get our next state, after taking action

            for edge_server in EdgeServer.all():
                edge_server_cpu = edge_server.cpu
                edge_server_memory = edge_server.memory
                edge_server_disk = edge_server.disk
                power = (edge_server_cpu * edge_server_memory * edge_server_disk) ** (1 / 3)
                vector = [edge_server_cpu, edge_server_memory, edge_server_disk, power]
                next_state_vector[network_switch_index[edge_server_dict[edge_server]]] = np.add(next_state_vector[network_switch_index[edge_server_dict[edge_server]]],vector)
                reward = reward + 1/edge_server.get_power_consumption() #Our reward is the inverse of the edge server's power consumption
                power = power + edge_server.get_power_consumption() #get the sum of powerconsumption of each edge server


            
            #Get our next state output from the graphical neural network
            next_state_vector_tensor  = torch.tensor(next_state_vector,dtype=torch.float64)
            adjacency_list_tensor = torch.tensor(matrix_array,dtype=torch.int64)

            next_state_vector_tensor = next_state_vector_tensor.to(model.lin1.weight.dtype)

            data = Data(x=next_state_vector_tensor, edge_index=adjacency_list_tensor)
            next_output = model(data.x, data.edge_index)

#            next_output = next_output.unsqueeze(0).unsqueeze(0)

            next_output = next_output.detach()
            
            #print(output.shape)
            #print(next_output.shape)
            
            
            #Use the next state to update the Deep Q network
            agent.update(output,action,next_output,reward,False)


            
            #Retrieve our Q network loss, and use it to update our GNN
            loss = agent.loss

            #print(loss)

            loss = loss.clone().detach()

            loss = criterion(torch.tensor(loss),torch.tensor(0))

            loss = torch.tensor(loss, requires_grad=True)


  #          print(loss)

#            loss = torch.tensor(loss,dtype=torch.float64,requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            total_power+=power #Sum our power consumption

    reward_list.append(total_reward)
    power_list.append(total_power) #Append power consumption to power list for plotting
    agent.epsilon*=agent.epsilon_decay #Reduce the probability of agent taking random action for exploration


            




            







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
simulator.initialize(input_file="sample_dataset3.json")

#Assigning the custom collect method
EdgeServer.collect = custom_collect_method



network_links_adj_list = {} #Our graph of adjacent network switches
network_switch_index = {} #network switch index mapping
matrix_list = list()
matrix_list.append(list())
matrix_list.append(list())

i = 0

for network_switch in NetworkSwitch.all(): #Initialise with empty list for each network switch, and assign index
    network_links_adj_list[network_switch] = list()
    network_switch_index[network_switch] = i
    i += 1



for network_link in NetworkLink.all(): #Iterate through network links and add the connected switches to our adjacency list
    print(network_link.nodes)
    network_links_adj_list[network_link.nodes[0]].append(network_link.nodes[1])
    network_links_adj_list[network_link.nodes[1]].append(network_link.nodes[0])
    matrix_list[0].append(network_switch_index[network_link.nodes[1]])
    matrix_list[1].append(network_switch_index[network_link.nodes[0]])



#dicts containing network switch to edge server mappings

network_switch_dict = {}
edge_server_dict = {}


#Map our edge serers and network switches

for network_switch in NetworkSwitch.all():
    print(network_switch.edge_servers)
    
    if(len(network_switch.edge_servers)):
        network_switch_dict[network_switch] = network_switch.edge_servers
        
        for edge_server in network_switch.edge_servers:
            edge_server_dict[edge_server] = network_switch

#Initialise our DQN agent and out CNN agent
agent = DQNAgent(4, len(EdgeServer.all()))
model = MyGNN(in_channels=4, out_channels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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

# Create a separate subplot for total power
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
plt.savefig('GCN_Q_learning_migration_power_consumption_final_uncropped.png')