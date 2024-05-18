import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

data = np.loadtxt('sin.csv', delimiter=',', unpack=True)
data = np.transpose(data)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(30, 40)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.fc3(x)
        return x


model = Model()
loss_fn = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.1)

inputs = torch.from_numpy(data[:, 0]).float().unsqueeze(1)
labels = torch.from_numpy(data[:, 1]).float().unsqueeze(1)
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=80, shuffle=True)
dataset_size = len(dataloader.dataset)
for epoch in range(2000):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Loop over batches in an epoch using DataLoader
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):

        y_batch_pred = model(x_batch)

        loss = loss_fn(y_batch_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Every 100 batches, print the loss for this batch
        # as well as the number of examples processed so far
        if id_batch % 100 == 0:
            loss, current = loss.item(), (id_batch + 1) * len(x_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")
# torch.save(model.state_dict(), 'sin_model.pth')

# Convert to TorchScript if needed
scripted_model = torch.jit.script(model)
scripted_model.save('sin_model.pt')
# scripted_model = torch.jit.trace(model, torch.Tensor(1))
# scripted_model.save('sin_model.pt')
