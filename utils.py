import gym, cv2, numpy as np
from collections import deque
from gym.spaces import Box
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch.nn as nn
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DuelingResNet(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        # Replace first conv to accept in_channels
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final fc
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
    def forward(self, x):
        x = x / 255.0
        feat = self.backbone(x).view(x.size(0), -1)
        value = self.value_stream(feat)
        adv   = self.adv_stream(feat)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q

def init_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # env = PreprocessFrame(env, (240, 256))
    env = PreprocessFrame(env, (84, 84))
    env = FrameStack(env, 4)
    return env

def plot_grayscale(state, path='grayscale.png'):
    plt.figure(figsize=(3,3))
    plt.axis("off")
    plt.imshow(state, cmap="gray", vmin=0, vmax=255)
    plt.savefig(path)
    plt.close()

def plot_scores(episodic_scores, average_scores, config, filename="train.png"):
    """
    Plot episodic and average scores, annotate with config, and save to file.
    """
    # 1) Create figure & axis with a bit more room up top
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2) Plot
    episodes = list(range(1, len(episodic_scores) + 1))
    ax.plot(episodes, episodic_scores, label="Episodic Score")
    ax.plot(episodes, average_scores,   label="Average 10 Episode Score")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left")
    
    # 3) Long title as suptitle so tight_layout won't clip it
    cfg_items = ", ".join(f"{k}={v}" for k, v in config.items())
    fig.suptitle(f"Training Scores | {cfg_items}", y=0.98, fontsize=12)
    
    # 4) Adjust layout, leaving headroom for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    # 5) Save
    fig.savefig(filename)
    plt.close(fig)

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84,84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(0,255, (shape[0],shape[1],1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(0,255, (shp[0],shp[1],shp[2]*k), dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.concatenate(self.frames, axis=2)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=2), reward, done, info
