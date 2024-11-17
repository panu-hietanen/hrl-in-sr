import torch
import torch.nn.functional as F
import torch.optim as optim

class TreeExpression:
    def __init__(self, data: torch.Tensor, target: torch.Tensor) -> None:
        self.data = data
        self.target = target
        self.expression: list[str] = None
        self.optimized_constants: torch.Tensor = None
        self.n_vars, self.n_samples = data.shape
        self.constants_optimized = False
        self.n_constants = 0

    def evaluate(self, expression: list[str]) -> torch.Tensor:
        """
        Evaluate the expression over batch data.
        """
        self.expression = expression.copy()
        if 'MAX' in self.expression:
            raise ValueError(f'Expression is not complete.')
        self._collect_constants()
        self.pos = 0
        if not self.constants_optimized and self.n_constants > 0:
            self._optimize_constants()
        self.pos = 0  # Reset position before evaluation
        result = self._evaluate_expression(self.optimized_constants)
        return result

    def _collect_constants(self) -> None:
        const_count = 0
        self.constants = []
        for idx, token in enumerate(self.expression):
            if token == 'C':
                const_name = f'C{const_count}'
                self.constants.append(const_name)
                self.expression[idx] = const_name
                const_count += 1
        self.n_constants = const_count

    def _evaluate_expression(self, constants: torch.Tensor) -> torch.Tensor:
        if self.pos >= len(self.expression):
            raise ValueError("Incomplete expression")

        token = self.expression[self.pos]
        self.pos += 1

        if token in ['+', '-', '*', '/', '^']:
            left = self._evaluate_expression(constants)
            right = self._evaluate_expression(constants)
            return self._apply_operator(token, left, right)
        elif token in ['sin', 'cos', 'exp', 'log']:
            operand = self._evaluate_expression(constants)
            return self._apply_operator(token, operand)
        elif token.startswith('C'):
            # Extract the index from the token
            index = int(token[1:])
            if index >= len(constants):
                raise ValueError(f"Constant index {index} out of bounds.")
            value = constants[index]
            # Expand value to match the number of samples
            return value.expand(self.n_samples)
        elif token.startswith('X'):
            # Variable value
            index = int(token[1:])
            if index >= self.data.shape[0]:
                raise ValueError(f"Variable index X{index} out of bounds.")
            return self.data[index]
        elif token == 'PAD':
            raise ValueError("Incomplete expression")
        else:
            try:
                value = float(token)
                return torch.full((self.n_samples,), value)
            except ValueError:
                raise ValueError(f"Unknown token: {token}")

    def _apply_operator(self, operator: str, *operands) -> torch.Tensor:
        try:
            if operator == '+':
                return operands[0] + operands[1]
            elif operator == '-':
                return operands[0] - operands[1]
            elif operator == '*':
                return operands[0] * operands[1]
            elif operator == '/':
                denom = operands[1]
                denom = torch.where(denom == 0, torch.tensor(1e-8, device=denom.device), denom)
                return operands[0] / denom
            elif operator == '^':
                return operands[0] ** operands[1]
            elif operator == 'sin':
                return torch.sin(operands[0])
            elif operator == 'cos':
                return torch.cos(operands[0])
            elif operator == 'exp':
                return torch.exp(operands[0])
            elif operator == 'log':
                arg = operands[0]
                arg = torch.where(arg <= 0, torch.tensor(1e-8, device=arg.device), arg)
                return torch.log(arg)
            else:
                raise ValueError(f"Unknown operator: {operator}")
        except Exception as e:
            print(f"Error applying operator '{operator}': {e}")
            return torch.full((self.n_samples,), float('inf'))

    def _optimize_constants(self) -> None:
        constants = torch.randn(self.n_constants, requires_grad=True)
        optimizer = optim.LBFGS([constants], max_iter=500)

        def closure():
            optimizer.zero_grad()
            self.pos = 0  # Reset position before evaluation
            y_pred = self._evaluate_expression(constants)
            loss = F.mse_loss(y_pred, self.target)
            loss.backward()
            return loss

        optimizer.step(closure)

        # After optimization, store the optimized constants
        self.optimized_constants = constants.detach().clone()
        self.constants_optimized = True

    def reset(self) -> None:
        self.expression = None
        self.optimized_constants = None
        self.constants_optimized = False
        self.n_constants = 0