import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

#create model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
#initalize model
model = Net(2,5,1) #input, hidden and output layer size
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.MSELoss() #loss function

model = Net(2,5,1)
print(model)

x = Variable(((torch.rand(10, 2) * 10)-10), requires_grad=True) #input data
y = Variable(((torch.rand(10, 1))), requires_grad=True) #output data



w = list(model.parameters())
a = []
for i in range(4): 
 ab = w[i].cpu().detach().numpy()
 ab0= ab.flatten()
 a = np.append(a,ab0)

#save weights and data
np.save("weights.npy",a)
np.save("input.npy", x.cpu().detach().numpy())
np.save("output.npy", y.cpu().detach().numpy())

#to calculate gradient
net_out = model(x)
loss = criterion(net_out, y)
loss.backward()

w = list(model.parameters())
print(w[0].grad)