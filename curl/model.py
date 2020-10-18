import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class D2RLNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, output_dim, num_layers=4):
        super(D2RLNetwork, self).__init__()

        in_dim = num_inputs+hidden_dim
        # Q1 architecture
        self.l1_1 = nn.Linear(num_inputs, hidden_dim)
        self.l1_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)

        self.out1 = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)

        self.num_layers = num_layers

    def forward(self, network_input):
        xu = network_input

        x1 = F.relu(self.l1_1(xu))        
        x1 = torch.cat([x1, xu], dim=1)
        
        x1 = F.relu(self.l1_2(x1))
        if not self.num_layers == 2:
            x1 = torch.cat([x1, xu], dim=1)
    
        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_4(x1))
            if not self.num_layers == 4:
                x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_6(x1))
            if not self.num_layers == 6:
                x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_8(x1))

        x1 = self.out1(x1)

        return x1


class D2RLGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers, action_space=None):
        super(D2RLGaussianPolicy, self).__init__()
        self.num_layers = num_layers
        
        in_dim = hidden_dim+num_inputs
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)


        if num_layers > 2:
            self.linear3 = nn.Linear(in_dim, hidden_dim)
            self.linear4 = nn.Linear(in_dim, hidden_dim)
        if num_layers > 4:
            self.linear5 = nn.Linear(in_dim, hidden_dim)
            self.linear6 = nn.Linear(in_dim, hidden_dim)
        if num_layers == 8:
            self.linear7 = nn.Linear(in_dim, hidden_dim)
            self.linear8 = nn.Linear(in_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = torch.cat([x, state], dim=1)

        x = F.relu(self.linear2(x))

        if self.num_layers > 2:
            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear3(x))

            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear4(x))

        if self.num_layers > 4:
            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear5(x))

            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear6(x))

        if self.num_layers == 8:
            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear7(x))

            x = torch.cat([x, state], dim=1)
            x = F.relu(self.linear8(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

