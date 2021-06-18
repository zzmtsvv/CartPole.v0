# NOT WORKING PROPERLY
''' Implementation of DDQN (CNN) via arxiv.org/pdf/1509.06461.pdf
    on super mario bros environment using pytorch
'''

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import random
import datetime
import time
import matplotlib.pyplot as plt
import os
import copy
from pathlib import Path
from collections import deque
import gym
from gym.wrappers import FrameStack
from gym.spaces import Box
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

env = JoypadSpace(env, COMPLEX_MOVEMENT)

env.reset()

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

class ArseNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    c, h, w = input_dim
    if h != 84:
      raise ArithmeticError(f"Expected input height: 84, got: {h}")
    if w != 84:
      raise ArithmeticError(f"Expected input width: 84, got: {w}")

    self.online = nn.Sequential(
        nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim)
    )
    self.target = copy.deepcopy(self.online)

    for theta in self.target.parameters():
      theta.requires_grad = False
  
  def forward(self, inputs, model):
    if model == 'online':
      return self.online(inputs)
    elif model == 'target':
      return self.target(inputs)

    
class Agent():
  def __init__(self, state_dim, action_dim, save_dir):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.save_dir = save_dir
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.memory = deque(maxlen=10000)
    self.batch_size = 32
    self.gamma = 0.9
    self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    self.burnin = 1e4  # min experiences before training
    self.learn_every = 3  # num of exp between updates to Q_online
    self.sync_every = 1e4  # num of experiences updates between Q_t and Q_o

    self.net = ArseNet(self.state_dim, self.action_dim).float()
    self.net = self.net.to(self.device)

    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=.00025)

    self.exploration_rate = 1
    self.exploration_rate_decay = 0.9999995
    self.exploration_rate_min = 0.1
    self.curr_step = 0

    self.save_every = 5e5

  
  def act(self, state):
    '''
    choose epsilon-greedy action given state and update the value of step
    '''
    # exploration
    if np.random.rand() < self.exploration_rate:
      action_idx = np.random.randint(self.action_dim)
    # exploitation
    else:
      state = state.__array__()
      state = torch.tensor(state).to(self.device)
      state = state.unsqueeze(0)
      action_values = self.net(state, model='online')
      action_idx = torch.argmax(action_values, axis=1).item()
    
    # decrease exploration_rate
    self.exploration_rate *= self.exploration_rate_decay
    if self.exploration_rate < self.exploration_rate_min:
      self.exploration_rate = self.exploration_rate_min
    
    self.curr_step += 1
    return action_idx
  
  def cache(self, state, next_state, action, reward, done):
    '''
    store the experience to self.memory
    '''
    state = state.__array__()
    next_state = next_state.__array__()
    state = torch.tensor(state).to(self.device)
    next_state = torch.tensor(next_state).to(self.device)
    action = torch.tensor([action]).to(self.device)
    reward = torch.tensor([reward]).to(self.device)
    done = torch.tensor([done]).to(self.device)

    self.memory.append((state, next_state, action, reward, done))
  
  def recall(self):
    '''
    Take a batch of experience from memory
    '''
    batch = random.sample(self.memory, self.batch_size)
    state, next_state, action, reward, done = map(torch.stack,
                                                  zip(*batch))
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
  
  def td_estimate(self, state, action):
    # the predicted optimal Q^*{online} for a given state
    current_Q = self.net(state, model="online")[np.arange(0, self.batch_size),
                                                action]
    return current_Q
  
  @torch.no_grad()
  def td_target(self, reward, next_state, done):
    '''
    TD Target - aggregation of current reward and the estimated Q
    in the next state
    '''
    next_state_Q = self.net(state, model='online')
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size),
                                                  best_action]
    return (reward + (1 - done.float()) * self.gamma * next_Q).float()
  
  def update_Q_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()
  
  def sync_Q_target(self):
    self.net.target.load_state_dict(self.net.online.state_dict())
  
  def save(self):
    num = int(self.curr_step // self.save_every)
    save_path = (self.save_dir / f"mario_net_{num}.chkpt")
    torch.save(dict(model=self.net.state_dict(),
                    exploration_rate=self.exploration_rate), save_path)
    print(f"ArseNet saved to {save_path} at step {self.curr_step}")
  
  def learn(self):
    if not self.curr_step % self.sync_every:
      self.sync_Q_target()
    if not self.curr_step % self.save_every:
      self.save()
    if self.curr_step < self.burnin:
      return None, None
    if self.curr_step % self.learn_every:
      return None, None
    
    # sample from memory
    state, next_state, action, reward, done = self.recall()

    # TD Estimate and TD Target
    td_est = self.td_estimate(torch.tensor(state), torch.tensor(action))
    td_trgt = self.td_target(torch.tensor(reward), torch.tensor(next_state),
                             torch.tensor(done))

    loss = self.update_Q_online(td_est, td_trgt)

    return td_est.mean().item(), loss

  
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

           
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n,
              save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate,
                      step=mario.curr_step)
