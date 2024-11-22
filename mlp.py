from os import system
import torch
from torch import Tensor
from torch.nn.functional import mse_loss

system("clear || cls")


class MLP:
    def __init__(
            self,
            layers_arr: list,
            m: int = 3,
            include_bias: bool = True,
            hidden_af: int = 1,
            out_af: int = 1,
            _device: str = "cpu"
    ) -> None:
        super().__init__()

        self.layers_arr = torch.tensor(layers_arr, device=_device)
        self._device = _device
        self.m = m
        self.include_bias = include_bias
        self.hidden_af = hidden_af
        self.out_af = out_af

    def __repr__(self) -> str:
        return f"MLP({self.layers_arr}, {self.include_bias}, {self.hidden_af}, {self.out_af})"

    def __call__(self, _x: Tensor) -> Tensor:
        _y_hat = _x
        layer_type = 0

        for idx, l in enumerate(self.layers_arr):
            w = torch.randn((l, self.m), device=self._device)
            b = torch.randn(1, device=self._device)

            # Output layer
            if idx == self.layers_arr.shape[0] - 1:
                layer_type = 1

            _y_hat = self.activation_func(_y_hat @ w.t() + b, layer_type=layer_type)

            self.m = _y_hat.shape[1]

        return _y_hat

    # By default, returns Linear activation
    # Layer_type holds which layer is using the activation function (0: Hidden Layers, 1: Output Layer)
    def activation_func(self, _x: Tensor, _af: int = 1, layer_type: int = 0) -> Tensor:
        _af = self.hidden_af

        if layer_type == 1:
            _af = self.out_af

        match _af:
            case 2:  # 2.Step
                return torch.heaviside(_x, torch.tensor(0.5))
            case 3:  # 3.ReLU
                return torch.relu(_x)
            case 4:  # 4.Sigmoid
                return torch.sigmoid(_x)
            case 5:  # 5.Softmax
                return torch.softmax(_x, dim=1, dtype=torch.float32)
            case _:  # 1.Linear
                return _x


if __name__ == "__main__":
    print("This is a MLP in order to calculate y_hat by Normal-Random Samples and Features\n")

    row = int(input("Enter the number of Samples:"))
    col = int(input("Enter the number of Features:"))

    layers = [int(i) for i in
              input("Enter the list of layers. Separate numbers by comma (E.X. 3, 5, 4, 3): ").split(',')]

    hidden_af = int(input(
        "Enter Activation Function for Hidden Layers\n\t1.Linear\n\t2.Step\n\t3.ReLU\n\t4.Sigmoid\n\t5.Softmax\n\tAF Num: "))
    out_af = int(input(
        "Enter Activation Function for Output Layers\n\t1.Linear\n\t2.Step\n\t3.ReLU\n\t4.Sigmoid\n\t5.Softmax\n\tAF Num: "))

    device = int(input(f"Enter Device:\n\t1.cpu\n\t2.cuda\n\tDevice: ")) or 1
    if device == 1:
        device = "cpu"
    else:
        device = "cuda"

    x = torch.randn((row, col), device=device)
    y = torch.randn((row, layers[-1]), device=device)

    mlp = MLP(layers, col, hidden_af=hidden_af, out_af=out_af, _device=device)
    y_hat = mlp(x)

    y_showed_rows = int(input("Enter the number of y and y_hat to print: "))

    system("clear || cls")
    print("y:\n", y[0:y_showed_rows], "\n")
    print("y_hat:\n", y_hat[0:y_showed_rows])

    # Calculate MSE
    mse = mse_loss(y_hat, y)
    print(f"\nMSE: {mse}\n")
