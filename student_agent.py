import gym
import torch
import cv2
import numpy as np
from train import DuelingResNet
from collections import deque
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingResNet(in_channels=4, n_actions=12).to(self.device)
        checkpoint = torch.load("checkpoints/policy_latest.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['policy_state_dict'])
        self.model.eval()
        self.frame_queue = deque(maxlen=4)
    
    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized
    
    def act(self, observation):
        frame = self.preprocess(observation)
        self.frame_queue.append(frame)
        
        while len(self.frame_queue) < 4:
            self.frame_queue.append(frame)
            
        stacked = np.stack(self.frame_queue, axis=-1)
        tensor = torch.tensor(stacked, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(tensor)
            action = q_values.argmax(dim=1).item()
        return action