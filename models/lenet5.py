import torch
import torch.nn.functional as F

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.W1 = torch.nn.Parameter(0.1*torch.randn(6,   1*5*5+1)) # CNN, (out_chs, in_chs*H*W + 1)
        self.W2 = torch.nn.Parameter(0.1*torch.randn(16,  6*5*5+1)) # CNN
        self.W3 = torch.nn.Parameter(0.1*torch.randn(16*4*4+1,120)) # FC
        self.W4 = torch.nn.Parameter(0.1*torch.randn(120+1,    84)) # FC
        self.W5 = torch.nn.Parameter(0.1*torch.randn(84+1,     10)) # FC
        
    def forward(self, x):
        x = F.conv2d(x, self.W1[:,:-1].view(6,1,5,5), bias=self.W1[:,-1])
        x = F.relu(F.max_pool2d(x, 2))
        x = F.conv2d(x, self.W2[:,:-1].view(16,6,5,5), bias=self.W2[:,-1])
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(x.view(-1, 16*4*4).mm(self.W3[:-1]) + self.W3[-1])
        x = F.relu(x.mm(self.W4[:-1]) + self.W4[-1])
        return x.mm(self.W5[:-1]) + self.W5[-1]