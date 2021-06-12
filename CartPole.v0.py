{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "name": "cartpolev0",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zzmtsvv/CartPole.v0/blob/main/CartPole.v0.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CAfnVXMF99k7"
      },
      "source": [
        "!apt-get install -y xvfb python-opengl > /dev/null 2>&1"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n_A3Jpp4COFY"
      },
      "source": [
        "!pip install gym pyvirtualdisplay > /dev/null 2>&1"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_afCy91gq60A"
      },
      "source": [
        "import gym\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython import display as ipythondisplay"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qCd5RJcsedpU"
      },
      "source": [
        "from pyvirtualdisplay import Display\n",
        "display = Display(visible=0, size=(400, 300))\n",
        "display.start()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eoaMJ2ctgCHE"
      },
      "source": [
        "import math\n",
        "import random\n",
        "import matplotlib\n",
        "from collections import namedtuple, deque\n",
        "from itertools import count\n",
        "from PIL import Image"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rqcix4PrGPo8"
      },
      "source": [
        "from pyvirtualdisplay import Display\n",
        "display = Display(visible=0, size=(400, 300))\n",
        "display.start()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2jxL-0M6q1VR"
      },
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "import torchvision.transforms as T"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hO3qdgk8rtSj"
      },
      "source": [
        "env = gym.make('CartPole-v0').unwrapped\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MowBMYtssKdm"
      },
      "source": [
        "random.seed(42)\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(device)\n",
        "plt.ion()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "K2P038RcsbNS"
      },
      "source": [
        "'''\n",
        "Replay memory stores transitions the agent observes. If we take samples from it\n",
        "randomly, batches are decorerlated\n",
        "'''\n",
        "\n",
        "Transition = namedtuple('Transition', ('state', 'action', 'next_state',\n",
        "                                       'reward'))\n",
        "\n",
        "class ReplayMemory(object):\n",
        "  def __init__(self, capacity):\n",
        "    self.memory = deque([], maxlen=capacity)\n",
        "\n",
        "  def push(self, *arguments):  # transition saving\n",
        "    self.memory.append(*arguments)\n",
        "  \n",
        "  def sample(self, batch_size):\n",
        "    return random.sample(self.memory, batch_size)\n",
        "  \n",
        "  def __len__(self):\n",
        "    return len(self.memory)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c8m7i57cvxJg"
      },
      "source": [
        "class ArseNet(nn.Module):  # привет Арс\n",
        "  '''\n",
        "  this cnn is trying to predict the expected return of taking each action given\n",
        "  the current input. (two outputs)\n",
        "  '''\n",
        "  def __init__(self, h, w, outputs):\n",
        "    super(ArseNet, self).__init__()\n",
        "    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)\n",
        "    self.bn1 = nn.BatchNorm2d(16)\n",
        "    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)\n",
        "    self.bn2 = nn.BatchNorm2d(32)\n",
        "    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)\n",
        "    self.bn3 = nn.BatchNorm2d(32)\n",
        "  \n",
        "    '''\n",
        "    num of linear input connections proportional to conv2d output which is\n",
        "    proportional to the image size\n",
        "    '''\n",
        "    \n",
        "    def conv2d_size_out(size, kernel_size=5, stride=2):\n",
        "      return (size - (kernel_size - 1) - 1) // stride  + 1\n",
        "    \n",
        "    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))\n",
        "    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))\n",
        "\n",
        "    linear_input = convw * convh * 32\n",
        "\n",
        "    self.head = nn.Linear(linear_input, outputs)\n",
        "\n",
        "  # called with either one element for determining the next action or \n",
        "  # with batch during optimization\n",
        "  def forward(self, x):\n",
        "    x = x.to(device)\n",
        "    x = F.relu(self.bn1(self.conv1(x)))\n",
        "    x = F.relu(self.bn2(self.conv2(x)))\n",
        "    x = F.relu(self.bn3(self.conv3(x)))\n",
        "\n",
        "    return self.head(x.view(x.size(0), -1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5WSunurX4zzf"
      },
      "source": [
        "input extraction"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YcAljzSs_qNK"
      },
      "source": [
        "from pyvirtualdisplay import Display\n",
        "\n",
        "\n",
        "display = Display(visible=False, size=(400, 300))\n",
        "display.start()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "e0rJDj8432Cr"
      },
      "source": [
        "resize = T.Compose([T.ToPILImage(),\n",
        "                    T.Resize(40, interpolation=Image.CUBIC),\n",
        "                    T.ToTensor()])\n",
        "\n",
        "def get_cart_location(screen_width):  # middle of cart\n",
        "  world_width = 2 * env.x_threshold\n",
        "  scale = screen_width / world_width\n",
        "  return int(env.state[0] * scale + screen_width / 2.0)\n",
        "\n",
        "def get_screen():\n",
        "  from pyvirtualdisplay import Display\n",
        "  display = Display(visible=0, size=(400, 300))\n",
        "  display.start()\n",
        "  # Transpose it into torch order (CHW)\n",
        "  screen = env.render(mode='rgb_array').transpose((2, 0, 1))\n",
        "\n",
        "  # Cart is in the lower half, so strip off the top and bottom of the screen\n",
        "  _, screen_h, screen_w = screen.shape\n",
        "  screen = screen[:, int(screen_h * 0.4):int(screen_h * 0.8)]\n",
        "  view_w = int(screen_w * 0.6)\n",
        "  cart_location = get_cart_location(screen_w)\n",
        "\n",
        "  if cart_location < view_w // 2:\n",
        "    slice_range = slice(view_w)\n",
        "  elif cart_location > screen_w - view_w // 2:\n",
        "    slice_range = slice(-view_width, None)\n",
        "  else:\n",
        "    slice_range = slice(cart_location - view_w // 2,\n",
        "                        cart_location + view_w // 2)\n",
        "  \n",
        "  screen = screen[:, :, slice_range]\n",
        "  screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0\n",
        "  screen = torch.from_numpy(screen)\n",
        "  # Resize, add a batch dimension (BCHW)\n",
        "  return resize(screen).unsqueeze(0)\n",
        "\n",
        "env.reset()\n",
        "plt.figure()\n",
        "plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy())\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "blp8biOmAAby"
      },
      "source": [
        "BATCH_SIZE = 128\n",
        "GAMMA = 0.999\n",
        "EPS_START = 0.9\n",
        "EPS_END = 0.05\n",
        "EPS_DECAY = 200\n",
        "TARGET_UPDATE = 10"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "w_iUuEaKEwa-"
      },
      "source": [
        "init_screen = get_screen()\n",
        "\n",
        "_, _, screen_height, screen_width = init_screen.shape"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QKDdFMsPSDUM"
      },
      "source": [
        "n_actions = env.action_space.n\n",
        "\n",
        "policy_net = ArseNet(screen_height, screen_width, n_actions).to(device)\n",
        "target_net = ArseNet(screen_height, screen_width, n_actions).to(device)\n",
        "\n",
        "target_net.load_state_dict(policy_net.state_dict())\n",
        "target_net.eval()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jd9YEXJFSjpm"
      },
      "source": [
        "optimizer = optim.RMSprop(policy_net.parameters())\n",
        "memory = ReplayMemory(10000)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9k75gzpuSwDB"
      },
      "source": [
        "steps_done = 0\n",
        "\n",
        "def select_action(state):\n",
        "  global steps_done\n",
        "  sample = random.random()\n",
        "  eps_threshold = EPS_END + (EPS_START - EPS_END) * \\\n",
        "        math.exp(-1. * steps_done / EPS_DECAY)\n",
        "  steps_done += 1\n",
        "\n",
        "  if sample > eps_threshold:\n",
        "    with torch.no_grad():\n",
        "      # x.max(1) returns largest column value for each row, second column\n",
        "      # on max result is index of max element, so\n",
        "      # we pick action with the larger expected reward.\n",
        "      return policy_net(state).max(1)[1].view(1, 1)\n",
        "  else:\n",
        "        return torch.tensor([[random.randrange(n_actions)]],\n",
        "                            device=device, dtype=torch.long)\n",
        "\n",
        "episode_durations = []\n",
        "\n",
        "def plot_durations():\n",
        "  plt.figure(2)\n",
        "  plt.clf() # clears current figure\n",
        "  durations_t = torch.tensor(episode_durations, dtype=torch.float)\n",
        "  plt.title('Training...')\n",
        "  plt.xlabel('episode')\n",
        "  plt.ylabel('duration')\n",
        "  plt.plot(durations_t.numpy())\n",
        "\n",
        "  # Take 100 episode averages and plot them too\n",
        "  if len(durations_t) >= 100:\n",
        "    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)\n",
        "    means = torch.cat(torch.zeros(99), means)\n",
        "    plt.plot(means.numpy())\n",
        "\n",
        "  plt.pause(0.01)\n",
        "  ipythondisplay.clear_output(wait=True)\n",
        "  ipythondisplay.display(plt.gcf()) # display current figure"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WONqLj-gYErY"
      },
      "source": [
        "def optimize_model():\n",
        "  if len(memory) < BATCH_SIZE:\n",
        "    return\n",
        "  transitions = memory.sample(BATCH_SIZE)\n",
        "  batch = Transition(*zip(*transitions))\n",
        "\n",
        "  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,\n",
        "                                          batch.next_state)),\n",
        "                                device=device, dtype=torch.bool)\n",
        "  non_final_next_states = torch.cat([s for s in batch.next_state\n",
        "                                                if s is not None])\n",
        "  state_batch = torch.cat(batch.state)\n",
        "  action_batch = torch.cat(batch.action)\n",
        "  reward_batch = torch.cat(batch.reward)\n",
        "\n",
        "  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the\n",
        "  # columns of actions taken. These are the actions which would've been taken\n",
        "  # for each batch state according to policy_net\n",
        "  state_action_values = policy_net(state_batch).gather(1, action_batch)\n",
        "\n",
        "  # Compute V(s_{t+1}) for all next states.\n",
        "  # Expected values of actions for non_final_next_states are computed based\n",
        "  # on the \"older\" target_net; selecting their best reward with max(1)[0].\n",
        "  # This is merged based on the mask, such that we'll have either the expected\n",
        "  # state value or 0 in case the state was final.\n",
        "\n",
        "  next_state_values = torch.zeros(BATCH_SIZE, device=device)\n",
        "  q = target_net(non_final_next_states).max(1)[0].detach()\n",
        "  next_state_values[non_final_mask] = q\n",
        "  # Compute the expected Q values\n",
        "  expected_state_action_values = (next_state_values * GAMMA) + reward_batch\n",
        "\n",
        "  # Huber Loss\n",
        "  criterion = nn.SmoothL1Loss()\n",
        "  loss = criterion(state_action_values,\n",
        "                   expected_state_action_values.unsqueeze(1))\n",
        "  \n",
        "  # optimize the model\n",
        "  optimizer.zero_grad()\n",
        "  loss.backward()\n",
        "  for parameter in policy_net.parameters():\n",
        "    parameter.grad.data.clamp_(-1, 1)\n",
        "  optimizer.step()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "jpV5PfnjhySo",
        "outputId": "20a2d710-bdea-40df-bb9e-2fc87ed091b9"
      },
      "source": [
        "num_episodes = 50\n",
        "\n",
        "for i_episode in range(num_episodes):\n",
        "  env.reset()\n",
        "  last_screen = get_screen()\n",
        "  current_screen = get_screen()\n",
        "\n",
        "  state = current_screen - last_screen\n",
        "  for t in count():\n",
        "    action = select_action(state)\n",
        "    _, reward, done, _ = env.step(action.item())\n",
        "    reward = torch.tensor([reward], device=device)\n",
        "\n",
        "    # observe new state\n",
        "    last_screen = current_screen\n",
        "    current_screen = get_screen()\n",
        "    if not done:\n",
        "      next_state = current_screen - last_screen\n",
        "    else:\n",
        "      next_state = None\n",
        "    \n",
        "    memory.push((state, action, next_state, reward))\n",
        "\n",
        "    state = next_state\n",
        "\n",
        "    optimize_model()\n",
        "    if done:\n",
        "      episode_durations.append(t + 1)\n",
        "      plot_durations()\n",
        "      break\n",
        "  # Update the target network, copying all weights and biases in DQN\n",
        "  if not i_episode % TARGET_UPDATE:\n",
        "    target_net.load_state_dict(policy_net.state_dict())\n",
        "\n",
        "print('Complete')\n",
        "env.render()\n",
        "env.close()\n",
        "plt.ioff()\n",
        "plt.show()\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 0 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GK7EUCvHlr-z"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}