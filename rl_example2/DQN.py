import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)


    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DQN(object):
    def __init__(
            self,
            num_actions,
            state_dim,
            device,
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            initial_eps=1,
            end_eps=0.001,
            eps_decay_period=25e4,
            eval_eps=0.001,
        ):

        self.device = device

        # Determine network type
        self.Q = FC_Q(state_dim, num_actions).to(self.device)
        # 冻结一份 Q
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Number of training iterations
        self.iterations = 0

        # 与环境交互的总步数（用来做 eps 衰减）
        self.total_steps = 0

    def _get_decay_eps(self):
        eps = self.initial_eps + self.slope * self.total_steps
        # 不让 eps 低于 end_eps
        eps = max(self.end_eps, eps)
        return eps
    

    def select_action(self, state, eval=False):
        # eps = self.eval_eps if eval else 0.05
        if eval:
            eps = self.eval_eps
        else:
            # 训练时使用随步数衰减的 epsilon
            eps = self._get_decay_eps()
            self.total_steps += 1

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)


    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            # target_Q = reward + not_done * self.discount * self.Q(next_state).max(1, keepdim=True)[0]
            # 使用target_Q
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss
        # Q_loss = F.smooth_l1_loss(current_Q, target_Q)
        Q_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        self.iterations += 1

        # 更新targetQ
        self.maybe_update_target()
        return {
            'Q': current_Q.mean().detach().cpu().numpy(),
            'Q_loss': Q_loss.mean().detach().cpu().numpy(),
        }


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())


    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
