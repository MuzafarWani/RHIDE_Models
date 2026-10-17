1. Install all the dependencies in the requirements.txt
2. Download and prepare the dataset by below command
	python scripts/prepare_cifar10_vgg16.py
3. Convert the dataset compact dataset file by below command
   python3 scripts/prepare_vgg16_val_npz.py
   
4. Then run the model with below command
   python scripts/train_federated_vgg16.py   --train_cfg configs_vgg16/train.yaml   --fed_cfg configs_vgg16/federated.yaml   --el_cfg configs_vgg16/energy_latency.yaml   --split_cfg configs_vgg16/split.yaml   --solver_cfg configs_vgg16/solver.yaml   --rl_cfg configs_vgg16/rl.yaml   --rounds 5
   
