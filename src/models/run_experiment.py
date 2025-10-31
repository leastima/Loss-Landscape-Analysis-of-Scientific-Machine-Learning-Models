# external libraries and packages
import wandb
import argparse
import sys
import traceback
import torch
import os
import json
import numpy as np

from src.models.train_utils import set_random_seed, train
from src.models.models import PINN

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_sample', type=int, default=1234, help='sample seed')
    parser.add_argument('--seed_init', type=int, default=1234, help='initial seed')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default='{"beta":30}', help='PDE coefficients')
    parser.add_argument('--opt', type=str, default='lbfgs',
                        help='optimizer to use')
    parser.add_argument('--opt_params', nargs='+', type=str,
                        default=None, help='optimizer parameters')
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--new_data', default=True, help='whether to create a new training set')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--save_path', type=str, default='../../output/', help='path to save the results of experiments')
    parser.add_argument('--save_model', default=True, help='Save the model for analysis later.')

    # Extract arguments from parser
    args = parser.parse_args()
    # set initial seed
    initial_seed = args.seed_init
    sample_seed = args.seed_sample

    # organize arguments for the experiment into a dictionary for logging purpose
    experiment_args = {
        "initial_seed": initial_seed,
        "pde": args.pde,
        "pde_params": args.pde_params,
        "opt": args.opt,
        "opt_params": args.opt_params,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "loss": args.loss,
        "num_x": args.num_x,
        "num_t": args.num_t,
        "num_res": args.num_res, 
        "epochs": args.epochs,
        "wandb_project": args.wandb_project,
        "new_data": args.new_data,
        "device": f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu',
        "save_path": args.save_path,
        "save_model": args.save_model
    }

    # print out arguments
    print("Seed set to: {}".format(initial_seed))
    print("Selected PDE type: {}".format(experiment_args["pde"]))
    print("Specified PDE coefficients: {}".format(
        experiment_args["pde_params"]))
    print("Optimizer to use: {}".format(experiment_args["opt"]))
    print("Specified optimizer parameters: {}".format(
        experiment_args["opt_params"]))
    print("Number of layers: {}".format(experiment_args["num_layers"]))
    print("Number of neurons per layer: {}".format(experiment_args["num_neurons"]))
    print("Number of spatial points (x): {}".format(experiment_args["num_x"]))
    print("Number of temporal points (t): {}".format(experiment_args["num_t"]))
    print("Number of random residual points to sample: {}".format(experiment_args["num_res"]))
    print("Number of epochs: {}".format(experiment_args["epochs"]))
    print("Weights and Biases project: {}".format(
        experiment_args["wandb_project"]))
    print("GPU to use: {}".format(experiment_args["device"]))

    pde_param = json.loads(experiment_args["pde_params"])
    if experiment_args["pde"] == 'convection':
        folder = os.path.join(experiment_args["save_path"], f'system_{experiment_args["pde"]}', 
                          f'N_f_{experiment_args["num_res"]}',f'beta_{float(pde_param["beta"])}')
        dataset_path = os.path.join("../../dataset", f'system_{experiment_args["pde"]}',
                          f'N_f_{experiment_args["num_res"]}',f'beta_{float(pde_param["beta"])}')
    elif experiment_args["pde"] == 'reaction':
        folder = os.path.join(experiment_args["save_path"], f'system_{experiment_args["pde"]}', 
                          f'N_f_{experiment_args["num_res"]}',f'rho_{float(pde_param["rho"])}')
        dataset_path = os.path.join("../../dataset", f'system_{experiment_args["pde"]}',
                          f'N_f_{experiment_args["num_res"]}',f'rho_{float(pde_param["rho"])}')

    with wandb.init(project=experiment_args["wandb_project"], config=experiment_args):
        # initialize model
        set_random_seed(initial_seed)
        model = PINN(in_dim=2, hidden_dim=experiment_args["num_neurons"], out_dim=1,
                     num_layer=experiment_args["num_layers"]).to(experiment_args["device"])
        # train the model
        try:
            train(model,
                  proj_name=experiment_args["wandb_project"],
                  pde_name=experiment_args["pde"],
                  pde_params=experiment_args["pde_params"],
                  loss_name=experiment_args["loss"],
                  opt_name=experiment_args["opt"],
                  opt_params_list=experiment_args["opt_params"],
                  n_x=experiment_args["num_x"],
                  n_t=experiment_args["num_t"],
                  n_res=experiment_args["num_res"],
                  num_epochs=experiment_args["epochs"],
                  device=experiment_args["device"],
                  folder=folder,
                  dataset_path=dataset_path,
                  new_data=experiment_args["new_data"],
                  sample_seed=sample_seed
                  )
        # log error and traceback info to W&B, and exit gracefully
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e
        
        if experiment_args["save_model"]:
            if experiment_args["pde"] == "convection":
                path = os.path.join("saved_models", f"system_{experiment_args['pde']}", f"N_f_{experiment_args['num_res']}",
                f"beta_{experiment_args['pde_params'][1]}")
            elif experiment_args["pde"] == "reaction":
                path = os.path.join("saved_models", f"system_{experiment_args['pde']}", f"N_f_{experiment_args['num_res']}",
                f"rho_{experiment_args['pde_params'][1]}")
            if not os.path.exists(path):
                os.makedirs(path)
            save_path = os.path.join(path, f"sample_{sample_seed}_init_{initial_seed}.pt")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()