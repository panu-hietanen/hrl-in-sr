from libs.tree import Tree, Node
from libs.expression import TreeExpression
import torch
import torch.nn as nn
import torch.nn.functional as F

class SREnv:
    def __init__(self, library: dict[str, int], data: torch.Tensor, target: torch.Tensor, max_length: int = 10) -> None:
        self.library = library
        self.action_symbols = list(self.library.keys())
        self.tree = Tree(self.library)
        self.expression = TreeExpression(data, target)
        self.done = False
        self.max_length = max_length
        self.n_vars, self.n_samples = data.shape

        self.leaves = [idx for idx, value in enumerate(self.library.values()) if value == 0]
        self.trig_symbols = [idx for idx, key in enumerate(self.library.keys()) if key == 'sin' or key == 'cos']

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
            reward = 0
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
    
    def get_action_mask(self) -> torch.Tensor:
        mask = torch.ones(len(self.library))
        expression = self.tree.encode(self.max_length)
        expression = [i for i in expression if i != "PAD" and i != "EOS"]
        if not expression:
            mask[self.leaves] = 0
            mask[self.trig_symbols] = 0
        elif expression[-1] == 'sin' or expression[-1] == 'cos':
            mask[self.trig_symbols] = 0
        if mask.sum() == 0:
            raise ValueError('No valid actions given.')
        return mask.unsqueeze(0)
