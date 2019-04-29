import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import pandas as pd
import pickle
import statistics

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPISODES = 2000
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
        self.affine1 = nn.Linear(self.state_space, 16)
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(16, self.action_space)

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
    # to solve "some of the strides of a given numpy array are negative"
    r = r - np.zeros_like(r)
    return r  # - r.mean()


def select_action(policy_estimator, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = state.to(device)
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
global_arr1 = []
global_arr2 = []


def s_factor(observations, current_estimation, shrinkage_point):
    if len(observations) > 1:
        Cov = np.cov([t.cpu().numpy() for t in observations], rowvar=False)
        Cov = np.multiply(Cov, np.identity(Cov.shape[0]))
        Cov_inv = np.linalg.inv(Cov + 0.00001 * np.random.rand(Cov.shape[0], Cov.shape[1]))
        # Cov_inv= Cov_inv.to(device)
        temp = torch.matmul(
            torch.matmul((current_estimation - shrinkage_point).view(-1), torch.from_numpy(Cov_inv).float().to(device)),
            (current_estimation - shrinkage_point))
        temp_arr.append(temp.item())
        # alpha = 1 - (list(current_estimation.size())[0] - 2) / temp
        alpha = 1 - (100 / temp)
        return alpha
        # return torch.max(torch.zeros_like(alpha), alpha)
    else:
        return 0


def l2_gradeint_debug(gradient):
    grand_truth = torch.load('grandT.pt')
    b = torch.zeros_like(grand_truth)
    return torch.dist(grand_truth, b, 2).item()


def cosine_gradeint_debug(gradient):
    grand_truth = torch.load('grandT.pt')
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(grand_truth, gradient).item()


def reinforce(env, policy_estimator, num_episodes=NUM_EPISODES,
              batch_size=BATCH_SIZE, gamma=0.99, js=False):
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_gradients = []
    grads_log = []

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
                batch_rewards.append(torch.FloatTensor(discount_rewards(rewards, gamma)))
                batch_states.append(torch.FloatTensor(states))
                batch_actions.append(torch.LongTensor(actions))
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    # state_tensor = torch.FloatTensor(batch_states)
                    # reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    # action_tensor = torch.LongTensor(batch_actions)

                    # Calculate loss
                    selected_logprobs = []
                    for i in range(len(batch_rewards)):
                        batch_states[i] = batch_states[i].to(device)
                        logprob = policy_estimator(batch_states[i])
                        batch_rewards[i] = batch_rewards[i].to(device)
                        batch_actions[i] = batch_actions[i].to(device)
                        selected_logprobs.append((batch_rewards[i] * logprob[np.arange(len(batch_actions[i])) ,batch_actions[i]]).sum())

                    # logprob = policy_estimator(state_tensor)
                    # baseline_tensor = (logprob[np.arange(len(action_tensor)), action_tensor]^2 * reward_tensor)
                    # / (logprob[np.arange(len(action_tensor)), action_tensor]^2)

                    # selected_logprobs = reward_tensor * \
                    #                     logprob[np.arange(len(action_tensor)), action_tensor]

                    loss = -sum(selected_logprobs)/len(selected_logprobs)

                    # Calculate gradients

                    loss.backward()

                    if js:
                        gw = smoothed_gradient(batch_gradients, .2)
                        batch_gradients.append(flatten_params(policy_estimator))
                        g_JS = gw + s_factor(batch_gradients, batch_gradients[-1], gw) * (
                                    flatten_params(policy_estimator) - gw)
                        grads_log.append(cosine_gradeint_debug(g_JS))
                        load_params(policy_estimator, g_JS)

                    else:
                        # print(cosine_gradeint_debug(flatten_params(policy_estimator)))
                        grads_log.append(cosine_gradeint_debug(flatten_params(policy_estimator)))

                    # l2_gradeint_debug(batch_gradients[-1], g_JS)

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
    # plt.plot(arr1, 'r')
    # plt.plot(arr2, 'b')
    # plt.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    # plt.set_xlabel('epoch')
    # plt.set_ylabel('difference with true gradient')
    # plt.show()
    #
    # global_arr1.append(arr1)
    # global_arr2.append(arr2)
    return total_rewards, grads_log


def draw_results(results):

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])

    window = len(results)

    rolling_mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    # rolling_mean = pd.Series(results).rolling(window).mean()
    # std = pd.Series(results).rolling(window).std()
    ax1.plot(rolling_mean, label="baseline")
    ax1.fill_between(range(len(results[0])), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)

    ax1.set_title('Episode Length Moving Average ({}-runs)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')
    ax1.legend(loc='upper left')


    window2 = int(50)
    rolling_mean2 = pd.Series(results[0]).rolling(window2).mean()
    std2 = pd.Series(results[0]).rolling(window2).std()
    ax2.plot(rolling_mean2, label="first run")
    ax2.fill_between(range(len(results[0])), rolling_mean2 - std2, rolling_mean2 + std2, color='orange',
                     alpha=0.2)
    ax2.set_title('Episode Length Moving Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.legend(loc='upper left')

    fig.tight_layout(pad=2)
    message = f"NUM_EPISODES: {NUM_EPISODES}, LEARNING_RATE: {LEARNING_RATE}, Num_RUNS: {Num_RUNS}, BATCH_SIZE: {BATCH_SIZE}"
    fig.text(.0, .0, message)

    plt.show()


def draw_compare_results(results, results1):

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])

    window = len(results)

    rolling_mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    rolling_mean1 = np.mean(results1, axis=0)
    std1 = np.std(results1, axis=0)
    # rolling_mean = pd.Series(results).rolling(window).mean()
    # std = pd.Series(results).rolling(window).std()
    ax1.plot(rolling_mean, label='JS', color='blue')
    ax1.fill_between(range(len(results[0])), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2, label='js_std')

    ax1.plot(rolling_mean1, label="MC", color='red')
    ax1.fill_between(range(len(results[0])), rolling_mean1 - std1, rolling_mean1 + std1, color='green',
                     alpha=0.2,label='mc_std')
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')
    ax1.legend(loc='upper left')

    window2 = int(50)
    rolling_mean2 = pd.Series(results[0]).rolling(window2).mean()
    std2 = pd.Series(results[0]).rolling(window2).std()
    ax2.plot(rolling_mean2)
    ax2.fill_between(range(len(results[0])), rolling_mean2 - std2, rolling_mean2 + std2, color='orange',
                     alpha=0.2)
    ax2.set_title('Episode Length Moving Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.legend(loc='upper left')


    fig.tight_layout(pad=2)
    message = f"NUM_EPISODES: {NUM_EPISODES}, LEARNING_RATE: {LEARNING_RATE}, Num_RUNS: {Num_RUNS}, BATCH_SIZE: {BATCH_SIZE}"
    fig.text(.0, .0, message)
    plt.show()


def compare_performance_grads(per_JS, per_MC, grad_JS, grad_MC):

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])

    window = len(per_JS)

    rolling_mean = np.mean(per_JS, axis=0)
    std = np.std(per_JS, axis=0)

    rolling_mean1 = np.mean(per_MC, axis=0)
    std1 = np.std(per_MC, axis=0)
    # rolling_mean = pd.Series(results).rolling(window).mean()
    # std = pd.Series(results).rolling(window).std()
    ax1.plot(rolling_mean, label='JS', color='blue')
    ax1.fill_between(range(len(per_JS[0])), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2, label='js_std')

    ax1.plot(rolling_mean1, label="MC", color='red')
    ax1.fill_between(range(len(per_MC[0])), rolling_mean1 - std1, rolling_mean1 + std1, color='green',
                     alpha=0.2 ,label='mc_std')
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')
    ax1.legend(loc='upper left')

    rolling_mean2 = np.mean(grad_JS, axis=0)
    std2 = np.std(grad_JS, axis=0)

    rolling_mean3 = np.mean(grad_MC, axis=0)
    std3 = np.std(grad_MC, axis=0)
    ax2.plot(rolling_mean2, label='JS', color='blue')
    ax2.fill_between(range(len(grad_JS[0])), rolling_mean2 - std2, rolling_mean2 + std2, color='orange',
                     alpha=0.2, label='js_std')

    ax2.plot(rolling_mean3, label="MC", color='red')
    ax2.fill_between(range(len(grad_MC[0])), rolling_mean3 - std3, rolling_mean3 + std3, color='green',
                     alpha=0.2,label='mc_std')
    ax2.set_title('Gradients cosine similarity')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('L2 norm')
    ax2.legend(loc='upper left')


    # fig.tight_layout(pad=2)
    message = f"NUM_EPISODES: {NUM_EPISODES}, LEARNING_RATE: {LEARNING_RATE}, Num_RUNS: {Num_RUNS}, BATCH_SIZE: {BATCH_SIZE}"
    fig.text(.0, .0, message)
    plt.show()


run_rewards = []
run_grads_mc = []

for run in range(Num_RUNS):
    pe = policy_estimator()
    pe = pe.to(device)
    returns, logs = reinforce(env, pe)
    run_rewards.append(returns)
    run_grads_mc.append(logs)
    # if run == 0:
    #     torch.save(flatten_params(pe), "grandT.pt")

run_rewards_js = []
run_grads_js = []
for run in range(Num_RUNS):
    pe = policy_estimator()
    pe = pe.to(device)
    returns, logs = reinforce(env, pe, js=True)
    run_rewards_js.append(returns)
    run_grads_js.append(logs)


draw_compare_results(run_grads_js,run_grads_mc)


print(len(run_grads_mc))
# print(len(run_grads_mc))
print(run_grads_mc)
# draw_compare_results(run_rewards_js, run_rewards)
# draw_compare_results(run_grads_js, run_grads_mc)
compare_performance_grads(run_rewards_js, run_rewards, run_grads_js, run_grads_mc)
# with open('outfile', 'wb') as fp:
#     pickle.dump(run_rewards, fp)
#
# with open ('outfile', 'rb') as fp:
#     results1 = pickle.load(fp)

# results1 = np.fromfile("a.bin", dtype=np.int64)
# draw_results(run_rewards, results1)

print(sum(temp_arr) / float(len(temp_arr)))
print(statistics.stdev(temp_arr))
