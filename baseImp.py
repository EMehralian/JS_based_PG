import argparse
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)
mc_grads_log = []


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 16)
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(16, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def flatten_params(model):
    return torch.cat([param.grad.data.view(-1) for param in model.parameters()], 0)


def l2_gradeint_debug(gradient):
    grand_truth = torch.load('grandT.pt')
    b = torch.zeros_like(gradient)
    return torch.dist(gradient, b, 2).item()


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i]
                  for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    r = r - np.zeros_like(r)
    return r


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, eps, eval):
    policy_loss = []
    returns = torch.tensor(policy.rewards)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    if eval:
        if mc_grads_log:
            mc_grads_log.append(mc_grads_log[-1].add(flatten_params(policy)))
        else:
            mc_grads_log.append(flatten_params(policy))
    else:
        optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def reinforce(policy, num_episode, eps, eval=False):
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10
    ep_return = []
    counter = 0
    for i_episode in range(1, num_episode):
        state, ep_reward = env.reset(), 0
        ep_rewards = []
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            # policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        r = discount_rewards(ep_rewards)
        policy.rewards.extend(r)

        ep_return.append(ep_reward)

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        counter += 1

        if counter % 1 == 0:
            finish_episode(policy, optimizer, eps, eval)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))

    return ep_return


def plot(result):
    mean = np.mean(result, axis=0)
    std = np.std(result, axis=0)

    plt.plot(result, label="baseline")
    # plt.fill_between(range(len(result[0])), mean - std, mean + std, color='orange',
    #                  alpha=0.2)
    plt.show()


def main():
    eps = np.finfo(np.float32).eps.item()
    NUM_RUN = 1
    run_return = []
    for i in range(NUM_RUN):
        policy = Policy()
        run_return.append(reinforce(policy, 250, eps, eval=False))
        reinforce(policy, 20000, eps, eval=True)
    normalized_grads = []
    for i in range(len(mc_grads_log)):
        normalized_grads.append(mc_grads_log[i].to(dtype=torch.float) / float(i + 1))

    torch.save(mc_grads_log[-1], "MC_grand_T.pt")
    estimated_grad = [l2_gradeint_debug(x) for x in normalized_grads]
    print(estimated_grad)
    plot(estimated_grad)


if __name__ == '__main__':
    main()

