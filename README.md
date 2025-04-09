## Usage

train pinn models and get the base metrics (training loss, test error, hessian...)
```sh
python main_pbc.py \
    --system convection \
    --beta=1 \
    --seed=0 \
    --collocation_seed 0 \
    --save_path ./results
```

cka metric:
```sh
python cka.py \
    --system convection \
    --beta=1 \
    --model_path $model_path \
    --model_seeds $model_seeds \
    --save_path ./results
```

lmc metric:
```sh
python mode_connectivity.py \
    --system convection \
    --beta=1 \
    --model_path $model_path \
    --model_seeds $model_seeds \
    --save_path ./results
```