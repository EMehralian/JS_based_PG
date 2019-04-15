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

# Hyperparameters
NUM_EPISODES = 1000
LEARNING_RATE = 0.01
Num_RUNS = 10
BATCH_SIZE = 10

env = gym.make('CartPole-v1')
env.seed(543)
torch.manual_seed(543)
s = env.reset()


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
        return F.log_softmax(action_scores, dim=1)


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r  # - r.mean()


def select_action(policy_estimator, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = torch.exp(policy_estimator(state))
    m = Categorical(probs)
    action = m.sample()
    return action.item()


def flatten_params(model):
    return torch.cat([param.grad.data.view(-1) for param in model.parameters()], 0)


def load_params(model, flattened):
    offset = 0
    for param in model.parameters():
        param.grad.data.copy_(flattened[offset:offset + param.nelement()].view(param.size()))
        offset += param.nelement()


def smoothed_gradient(gradients, gamma):
    # r = np.array([gamma ** i * gradients[i]
    #               for i in range(len(gradients))])
    # d = np.array([gamma ** i
    #               for i in range(len(gradients))])
    # return np.sum(r)/ np.sum(d)
    if gradients:
        return torch.mean(torch.stack(gradients))
    else:
        return 0


temp_arr = []


def s_factor(observations, current_estimation, shrinkage_point):
    if len(observations) > 1:
        Cov = np.cov([t.numpy() for t in observations], rowvar=False)
        # print(Cov)
        Cov_inv = np.linalg.inv(Cov + 0.00001 * np.random.rand(Cov.shape[0], Cov.shape[1]))
        temp = torch.matmul(
            torch.matmul((current_estimation - shrinkage_point).view(-1), torch.from_numpy(Cov_inv).float()),
            (current_estimation - shrinkage_point))
        temp_arr.append(temp)
        # alpha = 1 - (list(current_estimation.size())[0] - 2) / temp
        alpha = 1 - (10 / temp)
        return alpha
        # return torch.max(torch.zeros_like(alpha), alpha)
    else:
        return 0


def gradeint_debug(gradient):
    grand_truth = torch.load('grandT.pt')
    return torch.dist(gradient, grand_truth, 2)


def reinforce(env, policy_estimator, num_episodes=NUM_EPISODES,
              batch_size=BATCH_SIZE, gamma=0.99):
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_gradients = []
    JS_batch_gradients = []

    batch_counter = 0
    # Define optimizer
    optimizer = optim.Adam(pe.parameters(),
                           lr=LEARNING_RATE)

    # action_space = np.arange(env.action_space.n)
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

            action = select_action(policy_estimator, s_0)
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
                    logprob = policy_estimator(state_tensor)

                    # baseline_tensor = (logprob[np.arange(len(action_tensor)), action_tensor]^2 * reward_tensor)
                    # / (logprob[np.arange(len(action_tensor)), action_tensor]^2)

                    selected_logprobs = reward_tensor * \
                                        logprob[np.arange(len(action_tensor)), action_tensor]

                    loss = -selected_logprobs.mean()

                    # Calculate gradients

                    # gw = smoothed_gradient(batch_gradients, .2)
                    loss.backward()
                    # batch_gradients.append(flatten_params(policy_estimator))
                    # g_JS = gw + s_factor(batch_gradients, batch_gradients[-1], gw) * (
                    #             flatten_params(policy_estimator) - gw)
                    # JS_batch_gradients.append(g_JS)
                    # load_params(policy_estimator, g_JS)
                    # gradeint_debug(batch_gradients[-1], g_JS)
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 0

                # Print running average
                print("\rRun: {} Ep: {} Average of last 10: {:.2f}".format(run + 1,
                                                                           ep + 1, np.mean(total_rewards[-10:])),
                      end="")
    #
    # arr1 = [gradeint_debug(grad) for grad in JS_batch_gradients]
    # arr2 = [gradeint_debug(grad) for grad in batch_gradients]
    #
    # plt.plot(arr1,'r')
    # plt.plot(arr2,'b')
    # plt.show()

    return total_rewards


def draw_results(results):
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])

    window = len(results)

    rolling_mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    # rolling_mean = pd.Series(results).rolling(window).mean()
    # std = pd.Series(results).rolling(window).std()
    ax1.plot(std)
    # ax1.fill_between(range(len(results[0])), rolling_mean - std, rolling_mean + std, color='orange',
    #                  alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    window2 = int(50)
    rolling_mean2 = pd.Series(results[0]).rolling(window2).mean()
    std2 = pd.Series(results[0]).rolling(window2).std()
    ax2.plot(rolling_mean2)
    ax2.fill_between(range(len(results[0])), rolling_mean2 - std2, rolling_mean2 + std2, color='orange',
                     alpha=0.2)
    ax2.set_title('Episode Length Moving Average ({}-episode window)'.format(window2))
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    message = f"NUM_EPISODES: {NUM_EPISODES}, LEARNING_RATE: {LEARNING_RATE}, Num_RUNS: {Num_RUNS}, BATCH_SIZE: {BATCH_SIZE}"
    fig.text(.0, .0, message)
    plt.show()


run_rewards = []
for run in range(Num_RUNS):
    pe = policy_estimator()
    run_rewards.append(reinforce(env, pe))
    if run == 0:
        torch.save(flatten_params(pe), "grandT.pt")


draw_results(run_rewards)

print(np.mean(temp_arr))
print(np.std(temp_arr))
