import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, dropout_prob=0.5):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim     #Assign action dimension
        self.hidden_dim = hidden_dim     #assign dimension of hidden layer
        self.dropout_prob = dropout_prob   #Assign dropout probability

        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=(3, 1), stride=1)  #Convolutional layers
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), stride=1)
        self.fc1 = nn.Linear(48 * 16 * 4, hidden_dim)     #Flattening layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)      #Fully connected layers
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob)    #Dropout layers
        self.dropout2 = nn.Dropout(p=dropout_prob)

    def forward(self, state):
        x = state.unsqueeze(1)  # Add a channel dimension to the tensor
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout
        q_values = self.fc3(x)
        return q_values

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05, gamma=0.99, epsilon=1.0, epsilon_decay=0.8):
        self.state_dim = state_dim            #Assign state dimension, action dimension and hidden dimension
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr                          #Initialise hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy Q-network parameters to target network
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.01) #Use adam optimizer

        # Define loss function for Q-network
        self.criterion = nn.MSELoss()
        self.loss = None

    def choose_action(self, state):

        rand_number = np.random.rand()
#        print("Random Number : ",rand_number)
#        print("Epsilon : ",self.epsilon)
        if rand_number < self.epsilon:           #If our random number is less than our epsilon, perform random action for exploration
            action = np.random.randint(self.action_dim)
#            print("random action : ", action)
        else:
            with torch.no_grad():                   #Get the action with the maximum Q value
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
#                print("Q Value action : ", action)
        return action

    def update(self, state, action, next_state, reward, done, if_dynamic=False):
        

        state_tensor = torch.FloatTensor(state).unsqueeze(0)                #Convert numpy array to tensors
        action_tensor = torch.FloatTensor([action]).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0)

     #   print(action_tensor)
        # Update Q-network
        q_values = self.q_network(state_tensor)                    #Get Q values
        q_value = q_values.gather(1, action_tensor.long())
        next_q_values = self.target_network(next_state_tensor).detach()     #Get Q values for next state
        max_next_q_value = next_q_values.max(dim=1)[0]
        expected_q_value = reward_tensor + self.gamma * max_next_q_value * (1 - done_tensor)         #Calculate expected next Q value, and bacpropogate it through Q network
        loss = self.criterion(q_value, expected_q_value)
        self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())