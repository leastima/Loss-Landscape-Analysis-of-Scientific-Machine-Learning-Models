import wandb
import argparse
import sys
import traceback
import torch
import os
import numpy as np
import math

from src.models.train_utils import *
from src.models.models import PINN

class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n])
        I = torch.eye(n)
        H = I - unit / n
        H = H.cuda()
        return H @ K @ H

    def rbf(self, X, sigma=None):
        GX = torch.dot(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(
            self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma))
        )

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--num_x', type=int, default=257,
                        help='number of spatial sample points (power of 2 + 1)')
    parser.add_argument('--num_t', type=int, default=101,
                        help='number of temporal sample points')
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--save_path', type=str, help='path to save the results of experiments')

    # Extract arguments from parser
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    print("Collocation points: ", args.num_res)
    print("PDE coefficient: ", args.pde_params[1])

    model_list = []
    if args.pde == "convection":
        model_folder = os.path.join("saved_models", "system_convection", f"N_f_{args.num_res}", f"beta_{args.pde_params[1]}")
    elif args.pde == "reaction":
        model_folder = os.path.join("saved_models", "system_reaction", f"N_f_{args.num_res}", f"rho_{args.pde_params[1]}")

    for filename in os.listdir(model_folder):
        if filename.endswith(".pt"):
            path = os.path.join(model_folder, filename)
            model = PINN(in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers).to(device)
            model.load_state_dict(torch.load(path))
            model.eval()
            model_list.append(model)

    x_range, t_range, loss_func, pde_coefs = get_pde(args.pde, args.pde_params, args.loss)
    n_x_test = int((args.num_x - 1) / 2) + 1
    n_t_test = args.num_t
    x, t, data_params = get_data(x_range, t_range, n_x_test, n_t_test, random=False, device=device)

    metrics = {}
    
    # calculate the CKA similarity
    linear_cka_matrix = np.zeros((len(model_list), len(model_list)))
    for i in range(len(model_list)):
        for j in range(len(model_list)):
            if j <= i:
                linear_cka_matrix[i, j] = 0
            else:
                model0 = model_list[i]
                model1 = model_list[j]

                X = torch.vstack(predict(x, t, model0)).cpu().detach().numpy()
                Y = torch.vstack(predict(x, t, model1)).cpu().detach().numpy()

                X = torch.tensor(X).float().to(device)
                Y = torch.tensor(Y).float().to(device)

                np_cka = CKA()
                cka_res = np_cka.linear_CKA(X, Y)
                linear_cka_matrix[i, j] = cka_res

    metrics['linear_cka_matrix'] = linear_cka_matrix
    print("The CKA matrix is:")
    print(linear_cka_matrix)

    if args.pde == "convection":
        folder = os.path.join(args.save_path, "system_convection", f"N_f_{args.num_res}", f"beta_{float(args.pde_params[1])}")
    elif args.pde == "reaction":
        folder = os.path.join(args.save_path, "system_reaction", f"N_f_{args.num_res}", f"rho_{float(args.pde_params[1])}")
    if not os.path.exists(folder):
        os.makedirs(folder)
        np.save(folder + '/cka_metric.npy', metrics)
    else:
        np.save(folder + '/cka_metric.npy', metrics)

if __name__ == "__main__":
    main()
