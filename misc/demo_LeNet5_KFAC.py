import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
""" I used Thrandis's KFAC from url 
https://gist.github.com/Thrandis/9b3f75a130ec6c24a64117b7d9304c3f
"""
from kfac import KFAC

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=1000, shuffle=True)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(F.max_pool2d(self.conv1(x), 2))
        x = torch.tanh(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 256)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def test_loss(model, test_loader):
    model.eval()
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, dim=1)
            num_errs += torch.sum(pred!=target)
            
    return num_errs.item()/len(test_loader.dataset)


model = LeNet5()
preconditioner = KFAC(model, 0.001)
lr0 = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr0)
TrainLoss, TestLoss = [], []
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.nll_loss(output, target)
        for para in model.parameters():
            loss += 0.0002*torch.sum(para*para)
            
        TrainLoss.append(loss.item())
        loss.backward()
        preconditioner.step()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch: {}; batch: {}; train loss: {}'.format(epoch, batch_idx, TrainLoss[-1]))
    
    lr0 = (0.1**0.1)*lr0
    optimizer.param_groups[0]['lr'] = lr0
    TestLoss.append(test_loss(model, test_loader))
    print('Epoch: {}; best test loss: {}'.format(epoch, min(TestLoss)))