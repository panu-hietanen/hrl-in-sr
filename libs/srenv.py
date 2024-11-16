from lib.tree import Tree, Node
import math
import torch
from torch.nn import MSELoss

class SREnv:
    def __init__(self, library: dict[str, int], data, target, tree_depth: int = 10) -> None:
        self.library = library
        self.tree = Tree(self.library)
        self.done = False
        self.tree_depth = tree_depth

        self.target = target
        self.data = data

    def reset(self) -> list[Node]:
        self.tree = Tree(self.library)
        self.done = False
        return self.get_state()

    def step(self, action: str) -> tuple[list[Node], float, bool]:
        if action not in self.library:
            raise ValueError('Ensure the action is included in the library of symbols.')
        self.tree.add_action(action)
        if self.tree.complete():
            self.done = True
            reward = self.get_reward()
        else:
            reward = 0
        return self.tree.encode(self.tree_depth), reward, self.done
    
    def get_reward(self):
        return 1/ (1 + self.loss(self.tree.evaluate(self.data), self.target))
    
    def loss(self, data, target):
        return math.sqrt(data ** 2 - target ** 2)
        
    def get_state(self):
        return self.tree.encode(self.tree_depth)

