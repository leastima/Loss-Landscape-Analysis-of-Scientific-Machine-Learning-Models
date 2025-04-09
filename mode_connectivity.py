"""Run PINNs for convection/reaction/reaction-diffusion with periodic boundary conditions."""

import argparse
from net_pbc import *
import numpy as np
import os
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
import copy
from sklearn.manifold import MDS
from pyhessian import *

import math



################
# Arguments
################
parser = argparse.ArgumentParser(description='Characterizing/Rethinking PINNs')

parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--collocation_seed', type=int, default=0, help='Random initialization.')

parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
parser.add_argument('--rho', type=float, default=1.0, help='reaction coefficient for u*(1-u) term.')
parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

parser.add_argument('--visualize', default=False, help='Visualize the solution.')
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')

parser.add_argument('--save_path', type=str, help='path to save results.')
parser.add_argument('--model_path', type=str, help='path to load models to evaluate.')
parser.add_argument('--model_seeds', type=str, nargs='+', help='list of model seeds to evaluate.')

args = parser.parse_args()

# CUDA support

device = torch.device('cuda')

nu = args.nu
beta = args.beta
rho = args.rho

if args.system == 'diffusion': # just diffusion
    beta = 0.0
    rho = 0.0
elif args.system == 'convection':
    nu = 0.0
    rho = 0.0
elif args.system == 'rd': # reaction-diffusion
    beta = 0.0
elif args.system == 'reaction':
    nu = 0.0
    beta = 0.0

print('nu', nu, 'beta', beta, 'rho', rho)

# parse the layers list here
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

############################
# Process data
############################

x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

# remove initial and boundaty data from X_star
t_noinitial = t[1:]
# remove boundary at x=0
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

# sample collocation points only from the interior (where the PDE is enforced)
X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f, seed=args.collocation_seed)

if 'convection' in args.system or 'diffusion' in args.system:
    u_vals = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'rd' in args.system:
    u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'reaction' in args.system:
    u_vals = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
else:
    print("WARNING: System is not specified.")

u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
uu1 = Exact[0:1,:].T # u(x, t) at t=0
bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
uu2 = Exact[:,0:1] # u(-end, t)

# generate the other BC, now at x=2pi
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
bc_ub = np.hstack((x_bc_ub, t))

u_train = uu1 # just the initial condition
X_u_train = xx1 # (x,t) for initial condition

layers.insert(0, X_u_train.shape[-1])

############################
# Train the model
############################

set_seed(args.seed) # for weight initialization

model = PhysicsInformedNN_pbc(args.system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho,
                            args.optimizer_name, args.lr, args.net, args.L, args.activation, args.loss_style)
# model.train()


metrics = {}





# model 
path = f"{args.model_path}/seed_0.pt"

model_base = torch.load(path, map_location=device)

model_list = []
for seed in args.model_seeds:
    path = f"{args.model_path}/seed_{seed}.pt"

    model = torch.load(path, map_location=device)
    model.dnn.eval()
    if torch.is_grad_enabled():
        model.optimizer.zero_grad()
    model_list.append(copy.deepcopy(model.dnn))


# calculate the LMC
lmc_matrix = {}
for i in range(len(model_list)):
    lmc_matrix[i] = {}
    for j in range(len(model_list)):
        lmc_matrix[i][j] = {}

        model1 = model_list[i].state_dict()
        model2 = model_list[j].state_dict()
        # 初始化新 checkpoint 的状态字典
        new_state_dict = OrderedDict()

        error_list = []
        loss_list = []
        for alpha in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            # 计算线性加权平均
            for key in model1.keys():
                # 确保键在两个模型中一致
                if key in model2:
                    new_state_dict[key] = alpha * model1[key] + (1 - alpha) * model2[key]
                else:
                    raise ValueError(f"Key '{key}' not found in model2")

            model = copy.deepcopy(model_base)
            model.dnn.load_state_dict(new_state_dict)
            model.dnn.eval()


            # calculate the hessian eigenvalues and eigenvectors
            lmc_x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
            lmc_t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
            lmc_u = torch.tensor(u_train, requires_grad=True).float().to(device)

            lmc_x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
            lmc_t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)

            lmc_x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
            lmc_t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
            lmc_x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
            lmc_t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)

            if torch.is_grad_enabled():
                model.optimizer.zero_grad()
                
            u_pred = model.net_u(lmc_x_u, lmc_t_u)
            u_pred_lb = model.net_u(lmc_x_bc_lb, lmc_t_bc_lb)
            u_pred_ub = model.net_u(lmc_x_bc_ub, lmc_t_bc_ub)
            if nu != 0:
                u_pred_lb_x, u_pred_ub_x = model.net_b_derivatives(u_pred_lb, u_pred_ub, lmc_x_bc_lb, lmc_x_bc_ub)
            f_pred = model.net_f(lmc_x_f, lmc_t_f)

            loss_u = torch.mean((lmc_u - u_pred) ** 2)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            if nu != 0:
                loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.mean(f_pred ** 2)

            loss = loss_u + loss_b + args.L*loss_f
            loss_list.append(loss.item())


            # for test error
            u_pred = model.predict(X_star)
            error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
            error_list.append(error_u_relative)

        lmc_matrix[i][j]['loss'] = loss_list


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
np.save(args.save_path + '/cka_metric.npy', metrics)