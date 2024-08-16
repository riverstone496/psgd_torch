from models.lenet5 import LeNet5
from models.mlp import MLP

import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from preconditioned_stochastic_gradient_descent import Affine
import wandb
import numpy as np
from torch.utils.data.dataset import Subset

def train_loss(data, target, model):
    y = model(data)
    y = F.log_softmax(y, dim=1)
    return F.nll_loss(y, target)

def test_loss(test_loader, model):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = model(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred != target)
    return num_errs.item() / len(test_loader.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training')
    parser.add_argument('--train_size', type=int, default=-1, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--preconditioner_lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.2, help='learning rate')
    parser.add_argument('--model', type=str, default='mlp_tanh', help='number of epochs to train')
    parser.add_argument('--optim', type=str, default='psgd', help='number of epochs to train')
    parser.add_argument('--parametrization', type=str, default='sp', help='number of epochs to train')
    parser.add_argument('--width', type=int, default=1024, help='number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='learning rate')
    parser.add_argument('--preconditioner_init_scale', type=float, default=1.0, help='learning rate')
    args = parser.parse_args()
    wandb.init(config=args)
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    if args.train_size != -1:
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        train_idx = indices[:args.train_size]
        train_dataset = Subset(train_dataset, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,    
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])),    
        batch_size=args.test_batch_size, shuffle=False)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cuda'
    if args.model == 'lenet5':
        model = LeNet5()
    elif args.model == 'mlp_relu':
        model = MLP(n_hid = args.width)
    elif args.model == 'mlp_tanh':
        model = MLP(n_hid = args.width, nonlin=torch.tanh)

    opt = Affine(
        model.parameters(),
        # rank_of_approximation=rank,
        preconditioner_init_scale=args.preconditioner_init_scale,
        lr_params=args.lr,
        lr_preconditioner=args.preconditioner_lr,
        step_normalizer="1st", #this does Approx Newton on Lie Group
        momentum=args.momentum,   # match the momentum with Adam so that only their 'preconditioners' are different
        grad_clip_max_norm=10.0,
    )

    TrainLosses, best_test_loss, LogDets = [], 1.0, []
    for epoch in range(20):
        for _, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            def closure():
                xentropy = train_loss(data, target)
                l2 = sum(
                    [
                        torch.sum(torch.rand_like(p) * p * p)
                        for p in opt._params_with_grad
                    ]
                )
                return xentropy + args.weight_decay * l2, xentropy

            _, loss = opt.step(closure)
            TrainLosses.append(loss.item())

        best_test_loss = min(best_test_loss, test_loss())
        opt.lr_params *= (0.01) ** (1 / 19)
        opt.lr_preconditioner *= (0.01) ** (1 / 19)
        print(
            "Epoch: {}; best test classification error rate: {}".format(
                epoch + 1, best_test_loss
            )
        )