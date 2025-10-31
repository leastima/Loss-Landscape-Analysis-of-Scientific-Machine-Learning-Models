import wandb
import argparse
import sys
import traceback
import torch
import os
import numpy as np
import math
import copy

from src.models.train_utils import *
from src.models.models import PINN
from collections import OrderedDict

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

    parser.add_argument('--seed_sample', type=int, default=1234, help='sample seed')
    parser.add_argument('--hc', type=str, default='alm', help='hc method')

    # Extract arguments from parser
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    model_list = []
    pde_param = json.loads(args.pde_params)
    if args.pde == "convection":
        model_folder = os.path.join("../../saved_models", "system_convection", f"N_f_{args.num_res}",
                                    f"beta_{float(pde_param['beta'])}", args.hc)
        dataset_path = os.path.join("../../dataset", f'system_{args.pde}',
                          f'N_f_{args.num_res}', f"beta_{float(pde_param['beta'])}", args.hc, 'train_'+str(args.seed_sample)+'.pt')
    elif args.pde == "reaction":
        model_folder = os.path.join("../../saved_models", "system_reaction", f"N_f_{args.num_res}", f"rho_{float(pde_param['rho'])}", args.hc)
        dataset_path = os.path.join("../../dataset", "system_reaction", f"N_f_{args.num_res}", f"rho_{float(pde_param['rho'])}", args.hc, 'train_'+str(args.seed_sample)+'.pt')
    elif args.pde == "wave":
        model_folder = os.path.join("../../saved_models", "system_wave", f"N_f_{args.num_res}", f'beta_{float(pde_param["beta"])}_c_{float(pde_param["c"])}', args.hc)
        dataset_path = os.path.join("../../dataset", "system_wave", f"N_f_{args.num_res}", f'beta_{float(pde_param["beta"])}_c_{float(pde_param["c"])}', args.hc, 'train_'+str(args.seed_sample)+'.pt')
    elif args.pde == "reaction_diffusion":
        model_folder = os.path.join("../../saved_models", "system_reaction_diffusion", f"N_f_{args.num_res}", f'nu_{float(pde_param["nu"])}_rho_{float(pde_param["rho"])}', args.hc)
        dataset_path = os.path.join("../../dataset", "system_reaction_diffusion", f"N_f_{args.num_res}", f'nu_{float(pde_param["nu"])}_rho_{float(pde_param["rho"])}', args.hc, 'train_'+str(args.seed_sample)+'.pt')

    for filename in os.listdir(model_folder):
        if filename.endswith(".pt"):
            path = os.path.join(model_folder, filename)
            model = PINN(in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers).to(device)
            model.load_state_dict(torch.load(path))
            model.eval()
            model_list.append(model)

    # calculate the LMC
    lmc_matrix = {}
    for i in range(len(model_list)):
        lmc_matrix[i] = {}
        for j in range(len(model_list)):
            lmc_matrix[i][j] = {}

            model1 = model_list[i].state_dict()
            model2 = model_list[j].state_dict()
            new_state_dict = OrderedDict()

            error_list = []
            loss_list = []
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for key in model1.keys():
                    if key in model2:
                        new_state_dict[key] = alpha * model1[key] + (1 - alpha) * model2[key]
                    else:
                        raise ValueError(f"Key '{key}' not found in model2")
                    
                model = PINN(in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers).to(device)
                model.load_state_dict(new_state_dict)
                model.eval()

                x_range, t_range, loss_func, pde_coefs = get_pde(args.pde, args.pde_params, args.loss)
                data = torch.load(dataset_path, map_location=device)
                x = (data['x_res'], data['x_left'], data['x_upper'], data['x_lower'])
                t = (data['t_res'], data['t_left'], data['t_upper'], data['t_lower'])
                data_params = data['data_params']
            
                loss_res, loss_bc, loss_ic = loss_func(x, t, predict(x, t, model))
                loss = loss_res + loss_bc + loss_ic
                loss_list.append(loss.item())

                n_x_test = int((args.num_x - 1) / 2) + 1
                n_t_test = args.num_t
                x_test, t_test, data_params_test = get_data(x_range, t_range, n_x_test, 
                                                        n_t_test, random=False, device=device)

                with torch.no_grad():
                    predictions = torch.vstack(predict(x_test, t_test, model)).cpu().detach().numpy()
                targets = get_ref_solutions(args.pde, pde_coefs, x_test, t_test, data_params_test)
                test_l2re = l2_relative_error(predictions, targets)
                error_list.append(test_l2re)

            lmc_matrix[i][j]['loss'] = loss_list
            lmc_matrix[i][j]['error'] = error_list

    mc = np.zeros((len(model_list), len(model_list)))

    for i in range(len(model_list)):
        for j in range(len(model_list)):
            if j <= i:
                mc[i][j] = 0
            
            else:
                loss_list = lmc_matrix[i][j]['loss']  # loss list alpha from 0 to 1
        
                L_theta = loss_list[-1]  # alpha=1.0
                L_theta_prime = loss_list[0]  # alpha=0.0
        
                mid_losses = np.array(loss_list)
                midpoint = 0.5 * (L_theta + L_theta_prime)
        
                deviations = np.abs(midpoint - mid_losses)
                t_star_idx = np.argmax(deviations)
                L_gamma_t_star = mid_losses[t_star_idx]
        
                mc_val = midpoint - L_gamma_t_star
                mc[i][j] = mc_val

    print("The mc matrix is:")
    print(mc)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        np.save(model_folder + '/mode_connectivity.npy', mc)
    else:
        np.save(model_folder + '/mode_connectivity.npy', mc)

if __name__ == "__main__":
    main()
