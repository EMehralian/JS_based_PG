import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)


class policy_estimator(nn.Module):
    def __init__(self):
        super(policy_estimator, self).__init__()

        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.affine1 = nn.Linear(self.state_space, 128)
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, self.action_space)

        self.saved_log_probs = []
        self.rewards = []

        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

# class policy_estimator():
#     def __init__(self, env):
#         self.n_inputs = env.observation_space.shape[0]
#         self.n_outputs = env.action_space.n
#
#         # Define network
#         self.network = nn.Sequential(
#             nn.Linear(self.n_inputs, 16),
#             nn.ReLU(),
#             nn.Linear(16, self.n_outputs),
#             nn.Softmax(dim=-1))
#
#     def predict(self, state):
#         action_probs = self.network(torch.FloatTensor(state))
#         return action_probs


env = gym.make('CartPole-v1')
env.seed(543)
torch.manual_seed(543)
s = env.reset()
pe = policy_estimator()


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(pe.parameters(),
                           lr=0.01)

    action_space = np.arange(env.action_space.n)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            # action_probs = policy_estimator.predict(s_0).detach().numpy()
            # action = np.random.choice(action_space, p=action_probs)
            state = torch.from_numpy(s_0).float().unsqueeze(0)
            action_probs = policy_estimator(state)
            m = Categorical(action_probs)
            action = m.sample().item()
            # action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator(state_tensor))
                    selected_logprobs = reward_tensor * \
                                        logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

    return total_rewards


rewards = reinforce(env, pe)




window = int(50)
#
fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
rolling_mean = pd.Series(rewards).rolling(window).mean()
std = pd.Series(rewards).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(rewards)), rolling_mean - std, rolling_mean + std, color='orange',
                 alpha=0.2)
ax1.set_title('Episode Length Moving Average ({}-episode window)--V2--'.format(window))
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Length')

ax2.plot(rewards)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)

print("THE END")

plt.show()

#
# window = 10
# smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window
#                     else np.mean(rewards[:i+1]) for i in range(len(rewards))]
#
# plt.figure(figsize=(12,8))
# plt.plot(rewards)
# plt.plot(smoothed_rewards)
# plt.ylabel('Total Rewards')
# plt.xlabel('Episodes')
# plt.show()
