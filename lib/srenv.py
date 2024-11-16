from lib.tree import Tree
import torch
from torch.nn import MSELoss

class SREnv:
    def __init__(self, library, target):
        self.library = library
        self.tree = None
        self.loss = MSELoss()
        self.reset()

        self.target = target

    def reset(self):
        self.tree = Tree(self.library)
        self.done = False
        return self.get_state()

    def step(self, action):
        self.tree = self.tree.add_action(action)
        if self.tree.complete():
            self.done = True
            reward = self.get_reward()
        else:
            reward = 0
        return self.tree.encode(), reward, self.done
    
    def get_reward(self, y_pred):
        return 1/ (1 + self.loss(y_pred, self.target))
        
    def get_state(self):
        return (self.tree, self.done)

