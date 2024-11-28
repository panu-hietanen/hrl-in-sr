from libs.tree import Tree, Node
from libs.expression import TreeExpression
import torch
import torch.nn as nn
import torch.nn.functional as F

class SREnv:
    def __init__(self, library: dict[str, int], data: torch.Tensor, target: torch.Tensor, max_length: int = 10) -> None:
        self.library = library
        self.tree = Tree(self.library)
        self.expression = TreeExpression(data, target)
        self.done = False
        self.max_length = max_length
        self.n_vars, self.n_samples = data.shape

        self.target = target
        self.data = data

    def reset(self) -> list[str]:
        self.tree.reset()
        self.expression.reset()
        self.done = False
        return self.get_state()

    def step(self, action: str) -> tuple[list[str], float, bool]:
        if action not in self.library:
            raise ValueError('Ensure the action is included in the library of symbols.')
        self.tree.add_action(action)
        if self.tree.complete():
            self.done = True
            reward = self.get_reward()
        else:
            reward = self.estimate_reward()
        return self.tree.encode(self.max_length), reward, self.done
    
    def get_reward(self) -> float:
        expression = self.tree.encode(self.max_length)
        return self._squeeze(self.loss(self.expression.evaluate(expression), self.target))
    
    def loss(self, data: torch.Tensor, target: torch.Tensor) -> float:
        return F.mse_loss(data, target)
        
    def get_state(self) -> list[str]:
        return self.tree.encode(self.max_length)
    
    def _squeeze(self, x: float):
        return 1 / (1 + x)
    
    def estimate_reward(self) -> float:
        current_expression = self.tree.encode(self.max_length)
        nodes_needed = current_expression.count('PAD')
        evaluation = self.expression.approx_evaluate(current_expression, self.n_vars, nodes_needed)
        return self._squeeze(self.loss(evaluation, self.target)) / nodes_needed