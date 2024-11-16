import math

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

