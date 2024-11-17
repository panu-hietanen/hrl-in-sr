from libs.tree import Tree, Node, TreeExpression
import torch
import torch.nn as nn
import torch.nn.functional as F

class SREnv:
    def __init__(self, library: dict[str, int], data: torch.Tensor, target: torch.Tensor, max_depth: int = 10) -> None:
        self.library = library
        self.tree = Tree(self.library)
        self.expression = TreeExpression(data, target)
        self.done = False
        self.max_depth = max_depth

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
        return self.tree.encode(self.max_depth), reward, self.done
    
    def get_reward(self) -> float:
        expression = self.tree.encode(self.max_depth)
        return 1/ (1 + self.loss(self.expression.evaluate(expression), self.target))
    
    def loss(self, data: torch.Tensor, target: torch.Tensor) -> float:
        return F.mse_loss(data, target)
        
    def get_state(self) -> list[str]:
        return self.tree.encode(self.max_depth)

