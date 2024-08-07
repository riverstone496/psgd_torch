from models.lenet5 import LeNet5
from models.mlp import MLP

import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd
import wandb

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
    parser.add_argument('--batch_size', type=int, default=2048, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--model', type=str, default='mlp', help='number of epochs to train')
    parser.add_argument('--optim', type=str, default='psgd', help='number of epochs to train')
    parser.add_argument('--width', type=int, default=1024, help='number of epochs to train')
    args = parser.parse_args()
    wandb.init(config=args)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),    
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])),    
        batch_size=args.test_batch_size, shuffle=False)

    if args.model == 'lenet5':
        model = LeNet5()
    if args.model == 'mlp':
        model = MLP(n_hid = args.width)

    Qs = [[torch.eye(W.shape[0]), torch.eye(W.shape[1])] for W in model.parameters()]
    grad_norm_clip_thr = 0.1 * sum(W.numel() for W in model.parameters()) ** 0.5
    TrainLosses, best_test_loss = [], 1.0

    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = train_loss(data, target, model) + 1e-6 * sum([torch.sum(p * p) for p in model.parameters()])
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            if args.optim == 'psgd':
                vs = [torch.randn_like(W) for W in model.parameters()]
                Hvs = torch.autograd.grad(grads, model.parameters(), vs)
                with torch.no_grad():
                    Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, Hvs)]
                    pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
                    grad_norm = torch.sqrt(sum([torch.sum(g * g) for g in pre_grads]))
                    lr_adjust = min(grad_norm_clip_thr / grad_norm, 1.0)
                    [W.subtract_(lr_adjust * args.lr * g) for (W, g) in zip(model.parameters(), pre_grads)]
                    TrainLosses.append(loss.item())
                    if batch_idx % 3 == 0:
                        dgPdg = [dg.reshape(-1).T @ psgd.precond_grad_kron(Qlr[0], Qlr[1], dg).reshape(-1) for (Qlr, dg) in zip(Qs, Hvs)]
                        dxPinvdx = [dx.reshape(-1).T @ psgd.precond_grad_kron(torch.inverse(Qlr[0]), torch.inverse(Qlr[1]), dx).reshape(-1) for (Qlr, dx) in zip(Qs, vs)]
                        criterion = sum(dgPdg) + sum(dxPinvdx)
                        iteration = len(train_loader) * epoch + batch_idx
                        print(f"Epoch:{epoch+1} Iteration:{iteration}, Criteion:{criterion}")
                        wandb.log({"Epoch": epoch + 1, "Iteration":iteration, "Criteion": criterion, "TrainLoss":loss.item()})
            if args.optim == 'sgd':
                with torch.no_grad():
                    grad_norm = torch.sqrt(sum([torch.sum(g * g) for g in grads]))
                    lr_adjust = min(grad_norm_clip_thr / grad_norm, 1.0)
                    [W.subtract_(lr_adjust * args.lr * g) for (W, g) in zip(model.parameters(), grads)]
    
        test_err_rate = test_loss(test_loader, model)
        best_test_loss = min(best_test_loss, test_err_rate)
        args.lr *= (0.01) ** (1 / (args.epochs - 1))
        wandb.log({"Epoch": epoch + 1, "TestErrorRate": test_err_rate, "BestTestLoss":best_test_loss, "TestAcc":100*(1-test_err_rate)})
        print(f'Epoch: {epoch + 1}; best test classification error rate: {best_test_loss}')