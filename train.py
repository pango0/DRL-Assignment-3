import utils
import torchvision.models as models
import torch.nn as nn
import torch
import random
import numpy as np
import time
import os
from memory import PrioritizedReplayBuffer
from tqdm import tqdm

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

class Agent:
    def __init__(self, env, device):
        self.device = device
        self.policy_net = DuelingResNet(4, env.action_space.n).to(device)
        self.target_net = DuelingResNet(4, env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = PrioritizedReplayBuffer(100000)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)

    def optimize(self, batch_size, gamma):
        if len(self.memory) <= batch_size:
            return
        batch, idxs, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.tensor(states, device=self.device, dtype=torch.float32).permute(0,3,1,2)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32).permute(0,3,1,2)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

        Q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        best_next = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        Q_next = self.target_net(next_states).gather(1, best_next)
        Q_target = rewards.unsqueeze(1) + gamma * Q_next * (1 - dones.unsqueeze(1))

        td_errors = (Q_target - Q).detach().squeeze(1)
        loss = (weights * (Q_target - Q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(idxs, td_errors.abs().cpu().numpy())

        return loss.item()

    def save(self, filename=None):
        """Save policy_net (and optionally target_net) state dicts."""
        if filename is None:
            filename = f"policy_{int(time.time())}.pt"
        path = os.path.join('checkpoints', filename)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"=> Model checkpoint saved to {path}")
        return path
    
def train(config):
    num_episodes = config['num_episodes']
    device = config['device']
    epsilon = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    decay_rate = config['decay_rate']
    batch_size = config['batch_size']
    gamma = config['gamma']
    update_every = config['update_every']
    optimize_every = config['optimize_every']
    env = utils.init_env()
    agent = Agent(env, device)
    episodic_scores = []
    average_scores = []

    pbar = tqdm(range(1, num_episodes + 1), desc="Episodes", unit="ep", dynamic_ncols=True, leave=True)
    try:
        for episode in pbar:
            state = env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                state_tensor = torch.tensor(state, device=device, dtype=torch.float32)
                state_tensor = state_tensor.permute(2,0,1).unsqueeze(0)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = agent.policy_net(state_tensor).argmax(dim=1).item()

                next_state, reward, done, info = env.step(action)
                score += reward
                agent.memory.push((state, action, reward, next_state, done))
                state = next_state

                if (step + 1) % optimize_every == 0:
                    agent.optimize(batch_size, gamma)

                if (step + 1) % update_every == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

                step += 1
                
            epsilon = max(epsilon_end, epsilon * decay_rate)
            episodic_scores.append(score)
            avg_score = float(np.mean(episodic_scores))
            average_scores.append(avg_score)

            pbar.set_postfix({
                'score': score,
                'avg_score': avg_score
            })

            
            utils.plot_scores(episodic_scores, average_scores, config)
            if (episode + 1) % 100 == 0:
                agent.save(f"checkpoint_ep{episode}.pt")
                
    except KeyboardInterrupt:
        agent.save(f"checkpoint_ep{episode}.pt")
        
    finally:
        env.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = {
        'num_episodes': 1000,
        'device': device,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'decay_rate': 0.995,
        'batch_size': 256,
        'gamma': 0.99,
        'update_every': 1000,
        'optimize_every': 64
    }
    train(config)
