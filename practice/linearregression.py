import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):    ## in constructor we instantiate two nn.Linear module 
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #one in and one out
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

# construct our loss function and an Optimizer.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#training loop 
for epoch in range(500):
    #forward pass
    y_pred = model(x_data)

    #computer and print loss
    loss = criterion(y_pred,y_data)
    print(epoch, loss.data[0])

    #zero gradients, perform a backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#after training
hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])


