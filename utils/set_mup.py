import torch
import torch.nn as nn

def get_mup_setting(args):
    lrs = {}
    scaling_width = args.width / args.base_width
    if 'zero' in args.parametrization:
        sigma_last = 0
    else:
        sigma_last = 1
    return lrs, sigma_last

def initialize_weights(args, model, scale_last):
    """
    モデルの各層をHe初期化し、1層目と最終層の重みを指定された定数倍します。
    Args:
        model (nn.Sequential): PyTorchのSequentialモデル
        scale_first (float): 1層目の重みに対するスケーリング係数
        scale_last (float): 最終層の重みに対するスケーリング係数
    """
    for (name, module) in model.named_modules():
        initialize_layer(args, scale_last, name, module)

def initialize_layer(args, scale_last, name, layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        if args.activation == 'relu':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif args.activation == 'tanh':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')
        elif args.activation == 'sigmoid':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='sigmoid')
        elif args.activation == 'leaky_relu':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
        if 'output' in name:
            with torch.no_grad():
                print(layer, 'is multiplied by', scale_last)
                layer.weight.data.mul_(scale_last)