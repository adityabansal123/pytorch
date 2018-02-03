import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loop
for epoch in range(1000):

    #forward pass
    y_pred = model(x_data)

    #compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch,loss.data[0])

    #zero gradients, perform a backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#after training
hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours ", 7.0, model(hour_var).data[0][0] > 0.5)
