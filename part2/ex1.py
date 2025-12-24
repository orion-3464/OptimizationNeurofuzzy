import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.hidden = nn.Linear(1, 4)
        self.output = nn.Linear(4, 1)
        
        with torch.no_grad():
            self.hidden.weight = nn.Parameter(torch.tensor([[-1.0], [1.0], [1.0], [1.0]]))
            self.hidden.bias = nn.Parameter(torch.tensor([-2.0, 2.0, 1.0, 0.0]))
            
            self.output.weight = nn.Parameter(torch.tensor([[1.0, 1.0, -2.0, 2.0]]))
            self.output.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    model = CustomNN()
    x_test = torch.linspace(-5, 3, 100).view(-1, 1)
    y_pred = model(x_test).detach()

    plt.figure(figsize=(8, 5))
    plt.plot(x_test.numpy(), y_pred.numpy(), label='Neural Network Output', color='blue')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Αναπαράσταση της Συνάρτησης από το Νευρωνικό Δίκτυο")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()