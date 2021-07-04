  # based on pytorch tutorials from official website


import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import math
import random
import matplotlib
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

env = gym.make('CartPole-v0').unwrapped

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.ion()


Transition = namedtuple('Transition', ('state', 'action', 'next_state',
                                       'reward'))

class ReplayMemory(object):
  '''
  Replay memory stores transitions the agent observes. If we take samples from it
  randomly, batches are decorerlated
  '''
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *arguments):  # transition saving
    self.memory.append(*arguments)
  
  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
  
  def __len__(self):
    return len(self.memory)
  
  
  class ArseNet(nn.Module):  # привет Арс
  '''
  this cnn is trying to predict the expected return of taking each action given
  the current input. (two outputs)
  '''
  def __init__(self, h, w, outputs):
    super(ArseNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)
  
    '''
    num of linear input connections proportional to conv2d output which is
    proportional to the image size
    '''
    
    def conv2d_size_out(size, kernel_size=5, stride=2):
      return (size - (kernel_size - 1) - 1) // stride  + 1
    
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

    linear_input = convw * convh * 32

    self.head = nn.Linear(linear_input, outputs)

  # called with either one element for determining the next action or 
  # with batch during optimization
  def forward(self, x):
    x = x.to(device)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    return self.head(x.view(x.size(0), -1))
  
  
  resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):  # middle of cart
  world_width = 2 * env.x_threshold
  scale = screen_width / world_width
  return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
  # Transpose it into torch order (CHW)
  screen = env.render(mode='rgb_array').transpose((2, 0, 1))

  # Cart is in the lower half, so strip off the top and bottom of the screen
  _, screen_h, screen_w = screen.shape
  screen = screen[:, int(screen_h * 0.4):int(screen_h * 0.8)]
  view_w = int(screen_w * 0.6)
  cart_location = get_cart_location(screen_w)

  if cart_location < view_w // 2:
    slice_range = slice(view_w)
  elif cart_location > screen_w - view_w // 2:
    slice_range = slice(-view_width, None)
  else:
    slice_range = slice(cart_location - view_w // 2,
                        cart_location + view_w // 2)
  
  screen = screen[:, :, slice_range]
  screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
  screen = torch.from_numpy(screen)
  # Resize, add a batch dimension (BCHW)
  return resize(screen).unsqueeze(0)

n_actions = env.action_space.n

policy_net = ArseNet(screen_height, screen_width, n_actions).to(device)
target_net = ArseNet(screen_height, screen_width, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
  global steps_done
  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
  steps_done += 1

  if sample > eps_threshold:
    with torch.no_grad():
      # x.max(1) returns largest column value for each row, second column
      # on max result is index of max element, so
      # we pick action with the larger expected reward.
      return policy_net(state).max(1)[1].view(1, 1)
  else:
        return torch.tensor([[random.randrange(n_actions)]],
                            device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
  plt.figure(2)
  plt.clf() # clears current figure
  durations_t = torch.tensor(episode_durations, dtype=torch.float)
  plt.title('Training...')
  plt.xlabel('episode')
  plt.ylabel('duration')
  plt.plot(durations_t.numpy())

  # Take 100 episode averages and plot them too
  if len(durations_t) >= 100:
    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat(torch.zeros(99), means)
    plt.plot(means.numpy())

  plt.pause(0.01)
  ipythondisplay.clear_output(wait=True)
  ipythondisplay.display(plt.gcf()) # display current figure
 
def optimize_model():
  if len(memory) < BATCH_SIZE:
    return
  transitions = memory.sample(BATCH_SIZE)
  batch = Transition(*zip(*transitions))

  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)),
                                device=device, dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.

  next_state_values = torch.zeros(BATCH_SIZE, device=device)
  q = target_net(non_final_next_states).max(1)[0].detach()
  next_state_values[non_final_mask] = q
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  # Huber Loss
  criterion = nn.SmoothL1Loss()
  loss = criterion(state_action_values,
                   expected_state_action_values.unsqueeze(1))
  
  # optimize the model
  optimizer.zero_grad()
  loss.backward()
  for parameter in policy_net.parameters():
    parameter.grad.data.clamp_(-1, 1)
  optimizer.step()

  
num_episodes = 200

for i_episode in range(num_episodes):
  env.reset()
  last_screen = get_screen()
  current_screen = get_screen()

  plt.figure()
  plt.imshow(current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy())
  plt.show()

  state = current_screen - last_screen
  for t in count():
    action = select_action(state)
    _, reward, done, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)

    # observe new state
    last_screen = current_screen
    current_screen = get_screen()
    if not done:
      next_state = current_screen - last_screen
    else:
      next_state = None
    
    memory.push((state, action, next_state, reward))

    state = next_state

    optimize_model()
    if done:
      episode_durations.append(t + 1)
      plot_durations()
      break
  # Update the target network, copying all weights and biases in DQN
  if not i_episode % TARGET_UPDATE:
    target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
