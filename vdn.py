import argparse
import collections

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_WANDB = False  # if enabled, logs data on wandb server

class ReplayBuffer:
    def __init__(self, buffer_limit, n_agents, observation_space):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.n_agents = n_agents
        self.obs_size = observation_space[0].shape[0]

    def put(self, transition):
        """
        Parameters
        ----------
        transition: tuple(observations, actions, rewards, next_observations, done)
            observations: [n_agents, obs_size]
            actions: [n_agents]
            rewards: [n_agents]
            next_observations: [n_agents, obs_size]
            done: boolean
        """
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        """
        Parameters
        ----------
        batch_size: number of batch (batch is a group of chunk)
        chunk_size: number of samples per batch

        Returns
        -------
        observations, actions, rewards, next_observations, done
            observations: torch.Tensor[batch_size, chunk_size, n_agents, obs_size]
            actions: torch.Tensor[batch_size, chunk_size, n_agents]
            rewards: torch.Tensor[batch_size, chunk_size, n_agents]
            next_observations: torch.Tensor[batch_size, chunk_size, n_agents, obs_size]
            done: torch.Tensor[batch_size, chunk_size, 1]
        """
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        o_lst, a_lst, r_lst, o_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                o, a, r, o_prime, done = self.buffer[chunk_step]
                o_lst.append(o)
                a_lst.append(a)
                r_lst.append(r)
                o_prime_lst.append(o_prime)
                done_lst.append(done)

        return torch.tensor(o_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents, self.obs_size), \
            torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents), \
            torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents), \
            torch.tensor(o_prime_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents, self.obs_size), \
            torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, n_agents, observation_space, action_space):
        super(QNet, self).__init__()
        self.n_agents = n_agents
        self.hx_size = 32
        self.obs_size = observation_space[0].shape[0]
        self.n_acts = action_space[0].n
        self.nn_feature = nn.ModuleList([None for _ in range(self.n_agents)])
        self.nn_gru = nn.ModuleList([None for _ in range(self.n_agents)])
        self.nn_q = nn.ModuleList([None for _ in range(self.n_agents)])

        for i in range(n_agents):
            self.nn_feature.insert(i, nn.Sequential(nn.Linear(self.obs_size, 64),
                                                    nn.ReLU(),
                                                    nn.Linear(64, self.hx_size),
                                                    nn.ReLU(),
                                                    nn.GRUCell(self.hx_size, self.hx_size)))
            self.nn_gru.insert(i, nn.GRUCell(self.hx_size, self.hx_size))
            self.nn_q.insert(i, nn.Sequential(nn.Linear(self.hx_size, self.n_acts)))

    def forward(self, obs, hidden):
        """
        Parameters
        ----------
        obs: torch.Tensor[batch_size, n_agents, obs_size]
        hidden: torch.Tensor[batch_size, n_agents, self.hx_size]

        Returns
        -------
        q_values, hidden_states
            q_values: torch.Tensor[batch_size, n_agents, n_actions]
            hidden_states: torch.Tensor[batch_size, n_agents, self.hx_size]
        """
        batch_size = obs.shape[0]
        q_values = torch.empty(batch_size, self.n_agents, self.n_acts, dtype=torch.float)
        next_hidden = torch.empty(batch_size, self.n_agents, self.hx_size, dtype=torch.float)

        for agent_i in range(self.n_agents):
            x = self.nn_feature[agent_i](obs[:, agent_i, :])
            x = self.nn_gru[agent_i](x, hidden[:, agent_i, :])
            next_hidden[:, agent_i, :] = x
            q_values[:, agent_i, :] = self.nn_q[agent_i](x)

        return q_values, next_hidden

    def sample_action(self, obs, hidden, epsilon):
        """
        Parameters
        ----------
        obs: torch.Tensor[batch_size, n_agents, obs_size]
        hidden: torch.Tensor[batch_size, n_agents, self.hx_size]
        epsilon: With probability epsilon, choose a random action. Otherwise, choose greedy action

        Returns
        -------
        action, hidden
            action: torch.Tensor[batch_size, n_agents]
            hidden: torch.Tensor[batch_size, n_agents, self.hx_size]
        """
        batch_size = obs.shape[0]
        q_values, next_hidden = self.forward(obs, hidden)
        mask = torch.rand(batch_size) <= epsilon
        action = torch.empty(batch_size, self.n_agents)
        action[mask] = torch.randint(0, self.n_acts, action[mask].shape).float()
        action[~mask] = q_values[~mask].argmax(dim=2).float()
        return action, next_hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.n_agents, self.hx_size))

def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
    for _ in range(update_iter):
        o, a, r, o_prime, done = memory.sample_chunk(batch_size, chunk_size)
        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        loss = 0

        for step_i in range(chunk_size):
            q_values, hidden = q(o[:, step_i, :, :], hidden)
            q_a = q_values.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
            sum_q = q_a.sum(dim=1, keepdims=True)

            q_prime, _ = q(o_prime[:, step_i, :, :], hidden)
            q_prime_max = q_prime.argmax(dim=2, keepdims=True)
            q_target_prime, target_hidden = q_target(o_prime[:, step_i, :, :], target_hidden)
            q_target_prime_a = q_target_prime.gather(2, q_prime_max)
            sum_q_target_prime = q_target_prime_a.sum(dim=1)

            reward = r[:, step_i, :].sum(dim=1, keepdims=True) # (batch_size x 1)
            target = reward + gamma * sum_q_target_prime * (1 - done[:, step_i, :])

            loss += torch.norm(target.detach() - sum_q)

            done_mask = done[:, step_i].squeeze(-1).bool()
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()

def test(env, num_episodes, q):
    score = 0
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            while not all(done):
                action, hidden = q.sample_action(torch.tensor(state).unsqueeze(0), hidden, epsilon=0)
                next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist())
                score += sum(reward)
                state = next_state

    return score / num_episodes


def main(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes, max_epsilon, min_epsilon,
         test_episodes, warm_up_steps, update_iter, chunk_size, update_target_interval, recurrent):
    # create env.
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    memory = ReplayBuffer(buffer_limit, env.n_agents, env.observation_space)

    # create networks
    q = QNet(env.n_agents, env.observation_space, env.action_space)
    q_target = QNet(env.n_agents, env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    score = 0
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            while not all(done):
                action, hidden = q.sample_action(torch.tensor(state).unsqueeze(0), hidden, epsilon)
                action = action[0].data.cpu().numpy().tolist()
                next_state, reward, done, info = env.step(action)
                memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))
                score += sum(reward)
                state = next_state

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)

        if episode_i % update_target_interval:
            q_target.load_state_dict(q.state_dict())

        if (episode_i + 1) % log_interval == 0:
            test_score = test(test_env, test_episodes, q)
            train_score = score / log_interval
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, train_score, test_score, memory.size(), epsilon))
            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score, 'buffer-size': memory.size(),
                           'epsilon': epsilon, 'train-score': train_score})
            score = 0

    env.close()
    test_env.close()


if __name__ == '__main__':
    # torch.set_default_device('mps')
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Value Decomposition Network (VDN)')
    parser.add_argument('--env-name', required=False, default='ma_gym:Checkers-v0')
    parser.add_argument('--seed', type=int, default=1, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--max-episodes', type=int, default=15000, required=False)

    # Process arguments
    args = parser.parse_args()

    kwargs = {'env_name': args.env_name,
              'lr': 0.001,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'update_target_interval': 20,
              'log_interval': 100,
              'max_episodes': args.max_episodes,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
              'recurrent': not args.no_recurrent}

    if USE_WANDB:
        import wandb

        wandb.init(project='minimal-marl', config={'algo': 'vdn', **kwargs})

    main(**kwargs)
