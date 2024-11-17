import math
import torch
import torch.nn.functional as F
import torch.optim as optim

class Node:
    def __init__(self, symbol: str, arity: int) -> None:
        self.symbol = symbol
        if arity not in [0, 1, 2]:
            raise ValueError(f'Arity of symbol must be in 0-2.')
        self.arity = arity
        self.children: list[Node] = []

        self.is_leaf = self.arity == 0
        self.evaluated = None

    def complete(self):
        return len(self.children) == self.arity
    
    def add_child(self, child):
        if not self.complete():
            self.children.append(child)
        else:
            raise ValueError(f"Node '{self.symbol}' already has {self.arity} children.")
        
    def encode(self, depth: int, max_depth: int) -> list:
        nodes = []
        self._encoding(nodes, depth, max_depth)
        return nodes

    def _encoding(self, nodes: list, depth: int, max_depth: int) -> None:
        if depth > max_depth:
            nodes.append('PAD')
            return
        nodes.append(self.symbol)
        if self.is_leaf:
            return
        for child in self.children:
            child._encoding(nodes, depth + 1, max_depth)
        for _ in range(self.arity - len(self.children)):
            nodes.append('PAD')

    def evaluate(self, data: dict):
        if self.is_leaf:
            if 'X' in self.symbol:
                index = int(self.symbol.split('X')[1])
                return data[index]
            else:
                try:
                    return float(self.symbol)
                except ValueError:
                    raise ValueError(f"Unknown variable or constant: {self.symbol}")
        else:
            child_values = [child.evaluate(data) for child in self.children]
            try:
                if self.symbol == '+':
                    return child_values[0] + child_values[1]
                elif self.symbol == '-':
                    return child_values[0] - child_values[1]
                elif self.symbol == '*':
                    return child_values[0] * child_values[1]
                elif self.symbol == '/':
                    if child_values[1] == 0:
                        raise ZeroDivisionError("Division by zero")
                    return child_values[0] / child_values[1]
                elif self.symbol == 'sin':
                    return math.sin(child_values[0])
                elif self.symbol == 'cos':
                    return math.cos(child_values[0])
                elif self.symbol == 'exp':
                    return math.exp(child_values[0])
                elif self.symbol == 'log':
                    if child_values[0] <= 0:
                        raise ValueError("Logarithm of non-positive number")
                    return math.log(child_values[0])
                elif self.symbol == '^':
                    return child_values[0] ** child_values[1]
                else:
                    raise ValueError(f"Unknown operator: {self.symbol}")
            except Exception as e:
                raise ValueError(f"Error evaluating {self.symbol}: {e}")

class Tree:
    def __init__(self, library: dict[str, int]) -> None:
        self.root: Node = None
        self.current_nodes: list[Node] = []  # Stack to keep track of nodes needing children
        self.library = library

    def _create_node(self, action: str) -> Node:
        arity = self.library[action]
        return Node(action, arity)

    def add_action(self, action: str) -> None:
        if action not in self.library:
            raise ValueError('Ensure the action is included in the library of symbols.')
        node = self._create_node(action)
        if self.root is None:
            self.root = node
            if node.arity > 0:
                self.current_nodes.append(node)
        else:
            if self.complete():
                raise ValueError("Tree is already complete.")
            parent_node = self.current_nodes[-1]
            parent_node.add_child(node)
            if node.arity > 0:
                self.current_nodes.append(node)
            while self.current_nodes and self.current_nodes[-1].complete():
                self.current_nodes.pop()

    def complete(self) -> bool:
        return self.root is not None and not self.current_nodes
    
    def encode(self, max_depth: int) -> list:
        if self.root is None:
            return ['PAD'] * (2 ** max_depth - 1)
        encoding = self.root.encode(depth=0, max_depth=max_depth)
        return encoding
    
    def reset(self) -> None:
        self.root = None
        self.current_nodes = []

    def evaluate(self, variable_values):
        if not self.complete():
            raise ValueError("Cannot evaluate an incomplete tree")
        return self.root.evaluate(variable_values)

import numpy as np  # Or import torch if using PyTorch

class ExpressionEvaluator:
    def __init__(self, data: torch.Tensor, target: torch.Tensor):
        self.data = data
        self.target = target
        self.expression: list[str] = None
        self.constants = []
        self._collect_constants()
        self.n_vars, self.n_samples = data.shape

        self.constants_optimised = False

    def _collect_constants(self):
        const_count = 0
        for idx, token in enumerate(self.expression):
            if token == 'C':
                const_name = f'C{const_count}'
                self.constants.append(const_name)
                self.expression[idx] = const_name
                const_count += 1

    def evaluate(self, expression):
        """
        Evaluate the expression over batch data.
        :param variable_values: A NumPy array of shape (n_variables, n_samples)
        :param constant_values: A dictionary mapping constant names to their values
        :return: A NumPy array of shape (n_samples,)
        """
        self.expression = expression
        self.pos = 0
        if not self.constants_optimised:
            self._optimise_constants()
        result = self._evaluate_expression()
        return result

    def _evaluate_expression(self):
        if self.pos >= len(self.expression):
            raise ValueError("Incomplete expression")

        token = self.expression[self.pos]
        self.pos += 1

        if token in ['+', '-', '*', '/', '^']:
            left = self._evaluate_expression()
            right = self._evaluate_expression()
            return self._apply_operator(token, left, right)
        elif token in ['sin', 'cos', 'exp', 'log']:
            operand = self._evaluate_expression()
            return self._apply_operator(token, operand)
        elif token.startswith('C'):
            if token not in self.constant_values:
                raise ValueError(f"Value for constant '{token}' is not provided.")
            value = self.constant_values[token]
            return np.full(self.n_samples, value)
        elif token.startswith('X'):
            # Variable value
            index = int(token[1:])
            if index >= self.data.shape[0]:
                raise ValueError(f"Variable index X{index} out of bounds.")
            return self.data[index]
        else:
            try:
                value = float(token)
                return np.full(self.n_samples, value)
            except ValueError:
                raise ValueError(f"Unknown token: {token}")

    def _apply_operator(self, operator, *operands):
        try:
            if operator == '+':
                return operands[0] + operands[1]
            elif operator == '-':
                return operands[0] - operands[1]
            elif operator == '*':
                return operands[0] * operands[1]
            elif operator == '/':
                denom = operands[1]
                denom = np.where(denom == 0, 1e-8, denom)  # Avoid division by zero
                return operands[0] / denom
            elif operator == '^':
                return operands[0] ** operands[1]
            elif operator == 'sin':
                return np.sin(operands[0])
            elif operator == 'cos':
                return np.cos(operands[0])
            elif operator == 'exp':
                return np.exp(operands[0])
            elif operator == 'log':
                arg = operands[0]
                arg = np.where(arg <= 0, 1e-8, arg)  # Avoid log of non-positive numbers
                return np.log(arg)
            else:
                raise ValueError(f"Unknown operator: {operator}")
        except Exception as e:
            print(f"Error applying operator '{operator}': {e}")
            n_samples = operands[0].shape[0]
            return np.full(n_samples, np.inf)  # Return an array of inf values
        
    def _optimise_constants(self):
        n_constants = len(self.constants)
        constants = torch.randn(n_constants, requires_grad=True)
        optimiser = optim.LBFGS([constants], max_iter=100)

        def closure():
            optimiser.zero_grad()
            # Prepare constant values as a dictionary
            constant_values = {self.constants[i]: constants[i] for i in range(n_constants)}
            # Evaluate the expression
            y_pred = self.evaluate()
            # Compute the loss
            loss = F.mse_loss(y_pred, self.target)
            loss.backward()
            return loss
        
        optimiser.step(closure)

        optimized_constants = constants.detach().numpy()
        return optimized_constants





def optimize_constants(evaluator, X, y):
    n_constants = len(evaluator.constants)
    # Initialize constants as torch tensors with requires_grad=True
    constants = torch.randn(n_constants, requires_grad=True)
    
    # Convert X and y to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    optimizer = optim.LBFGS([constants], max_iter=100)
    
    def closure():
        optimizer.zero_grad()
        # Prepare constant values as a dictionary
        constant_values = {evaluator.constants[i]: constants[i] for i in range(n_constants)}
        # Evaluate the expression
        y_pred = evaluator.evaluate(X_tensor, constant_values)
        # Compute the loss
        loss = F.mse_loss(y_pred, y_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    # After optimization, get the optimized constants
    optimized_constants = constants.detach().numpy()
    return optimized_constants

