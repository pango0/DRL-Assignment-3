import torch
import cv2
import numpy as np
from collections import deque
from utils import DuelingCNN          # or DuelingResNet if you prefer
import utils
import gdown                       # leave here if you actually need to download the weights
# https://drive.google.com/file/d/1uHBw6_k69TOyzZ51Gh4B8xVbbYDqUesC/view?usp=sharing
gdown.download('https://drive.google.com/uc?id=1uHBw6_k69TOyzZ51Gh4B8xVbbYDqUesC', 'stage-1.pt')
gdown.download('https://drive.google.com/uc?id=1CI3bzlEynd3obMB5ZxiSzQHzqU4QcsQv', 'stage-2.pt')
# https://drive.google.com/file/d/1gh9I6WG3Rh88f49xqKidXSlDRVQjKwZx/view?usp=sharing
class Agent(object):
    """
    Mario agent that performs *internal* frame‑skipping:
    * Repeats the last chosen action for (skip‑1) calls to `act`.
    * Only every `skip`‑th call feeds a frame stack through the network.
    
    The external evaluation environment therefore receives an action
    on **every** call, so the input/output signature remains unchanged.
    """
    def __init__(self):
        self.step = 0
        # ----- model & device -------------------------------------------------
        self.device = "cpu"   # change to "cuda" if permitted
        self.stage_1_model  = DuelingCNN(in_channels=4, n_actions=12).to(self.device)
        self.stage_2_model  = DuelingCNN(in_channels=4, n_actions=12).to(self.device)
        checkpoint_1 = torch.load(
            "stage-1.pt",
            map_location=self.device,
            weights_only=True
        )
        checkpoint_2 = torch.load(
            "stage-2.pt",
            map_location=self.device,
            weights_only=True
        )
        self.stage_1_model.load_state_dict(checkpoint_1["policy_state_dict"])
        self.stage_2_model.load_state_dict(checkpoint_2["policy_state_dict"])
        self.stage_1_model.eval()
        self.stage_2_model.eval()
        
        self.model = self.stage_1_model
        self.stage2_at_step = 2002
        
        # self.stage_2_start_state = np.load("stage-2_state.npy")
        
        # ----- frame processing ----------------------------------------------
        self.frame_queue = deque(maxlen=4)    # holds last 4 pre‑processed frames
        
        # ----- skipping & action latch ---------------------------------------
        self.skip              = 4            # repeat each chosen action 4 times
        self._frames_left      = 0            # how many repeats still to do
        self._latched_action   = 0            # last action returned
        self.step              = 0            # debug / logging counter

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        """RGB → 84×84 grayscale uint8"""
        gray    = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------
    def act(self, observation):
        """
        Called once **per raw frame** by the evaluation harness.
        Returns one discrete action (int) every time.
        """
        if self.step == self.stage2_at_step:
            # ----- swap models
            self.model = self.stage_2_model      # ← all following calls use stage-2
            # ----- reset every per-episode buffer you care about
            self.frame_queue.clear()             # throw away old frames
            self._frames_left    = 0             # cancel any action skip window
            self._latched_action = 0
            print("[INFO] Switched to stage-2 model at step",
                  self.stage2_at_step, flush=True)
            
        self.step           += 1
        # ------------------------------------------------------------------
        # 1.  If we're inside a skip window, reuse the latched action
        # ------------------------------------------------------------------
        if self._frames_left > 0:
            self._frames_left -= 1
            return self._latched_action

        # ------------------------------------------------------------------
        # 2.  This is the first frame after the window → run the policy
        # ------------------------------------------------------------------
        frame = self.preprocess(observation)
        self.frame_queue.append(frame)
        
        # pad at episode start
        while len(self.frame_queue) < 4:
            self.frame_queue.append(frame)
        
        stacked = np.stack(self.frame_queue, axis=-1)                 # H×W×C
        tensor  = (
            torch.tensor(stacked, dtype=torch.float32, device=self.device)
                 .permute(2, 0, 1)                                    # C×H×W
                 .unsqueeze(0)                                        # B×C×H×W
        )
        
        with torch.no_grad():
            q_values = self.model(tensor)
            action   = int(q_values.argmax(dim=1).item())
        
        # ------------------------------------------------------------------
        # 3.  Latch the action & set counter for the next (skip‑1) frames
        # ------------------------------------------------------------------
        self._latched_action = action
        self._frames_left    = self.skip - 1

        return action
