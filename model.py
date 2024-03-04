import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 

device = ("cuda" if torch.cuda.is_available()
          else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, x_shape, y_shape):
        super().__init__()
        self.flatten = nn.Flatten()

        self.nn = nn.Sequential(
            nn.Linear(x_shape, 24),
            nn.Tanh(),
            nn.Linear(24, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, y_shape)
        )

        self.lossFn = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)

    def forward(self, x):
        x = self.flatten(x)
        output = self.nn(x)
        return output
        
    # def lossFunction(self, output, target):
    #     loss = nn.MSELoss()
    #     return loss(output, target)
        
def train(model, x, y):
    loss_list = []
    model.zero_grad() # Clears all accumulated gradients within the model tensors
    model.train() # activates backprop capabilities for tensors having requires_grad = True

    n_epochs = 2000 # Total Epochs
    batch_size = 50 # Split x (about 1700) into batches of 50

    print(np.shape(x), np.shape(y))
    
    for epoch in range(n_epochs):
        print(f"Epoch number : {epoch}")

        batch_index = torch.arange(0, len(x), batch_size)
        for start in batch_index:
            X = x[start:start+batch_size, :].to(device)
            Y = y[start:start+batch_size, :].to(device)

            pred = model(X)
            loss = model.lossFn(pred, Y)

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            if start == batch_index[-1]:
                print(f"loss: {loss:>4f}")
                loss_list.append(float(loss))

    plt.plot(loss_list)
    plt.show()

def test(model, x, y):
    model.eval()
    x_test = x.to(device)
    y_test = y.to(device)
    
    pred = model(x_test)
    test_loss = model.lossFn(pred, y_test)

    print(f"Test Loss = {test_loss}")

    
    





    

