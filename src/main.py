# main.py

import torch
from torch.utils.data import DataLoader
from client import get_client_fn
from evaluate import get_evaluate_fn
from models import CNNModel
from datasets import get_mnist
from attack import MaliciousAggregation

def main(num_clients, attack_round):
    SEED = 0
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test = get_mnist()

    num_train_samples = len(train)
    client_dataset_size = num_train_samples // num_clients
    remainder = num_train_samples % num_clients

    train_datasets = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + client_dataset_size + (1 if i < remainder else 0)
        train_subset = torch.utils.data.Subset(train, range(start_idx, end_idx))
        train_datasets.append(train_subset)
        start_idx = end_idx

    global_model = CNNModel().to(device)
    global_model_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    client_parameters = [global_model_state.copy() for _ in range(num_clients)]

    malicious_aggregator = MaliciousAggregation(attack_round=attack_round, malicious_client_idx=0)

    # Simulation loop
    for round_num in range(100):
        print(f"\nStarting federated round {round_num+1}")

        client_updates = []

        def train_client(client_idx):
            client_model = CNNModel().to(device)
            client_model.load_state_dict(client_parameters[client_idx])
            client_train_fn = get_client_fn(client_model, train_datasets)
            client = client_train_fn(client_idx)
            updated_params, num_samples, metrics = client.fit()
            return {
                'client_idx': client_idx,
                'params': updated_params,
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy']
            }

        for client_idx in range(num_clients):
            client_update = train_client(client_idx)
            client_updates.append(client_update)

        # Aggregate updates using the modified aggregation function
        q = 1.0  
        aggregated_update = malicious_aggregator.aggregate(round_num, client_updates, q=q)
        global_model.load_state_dict(aggregated_update)

        client_parameters = [aggregated_update.copy() for _ in range(num_clients)]

        print(f"Round {round_num+1} completed.")

    torch.save(global_model.state_dict(), f"global_model_{num_clients}_clients.pth")

if __name__ == "__main__":
    for attack_round in [0, 80]:
        for num_clients in [3, 10, 30]:
            print(f"\n=== Running simulation with {num_clients} clients and attack round at {attack_round} ===")
            main(num_clients, attack_round)


# main.py
# import torch
# from torch.utils.data import DataLoader
# from client import get_client_fn
# from models import CNNModel
# from datasets import get_mnist
# from attack import MaliciousAggregation

# def main(num_clients, attack_round):
#     SEED = 0
#     torch.manual_seed(SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load MNIST dataset
#     train, test = get_mnist()

#     # Split dataset among clients
#     num_train_samples = len(train)
#     client_dataset_size = num_train_samples // num_clients
#     remainder = num_train_samples % num_clients

#     train_datasets = []
#     start_idx = 0
#     for i in range(num_clients):
#         end_idx = start_idx + client_dataset_size + (1 if i < remainder else 0)
#         train_subset = torch.utils.data.Subset(train, range(start_idx, end_idx))
#         train_datasets.append(train_subset)
#         start_idx = end_idx

#     # Initialize the global model
#     global_model = CNNModel().to(device)
#     global_model_state = {k: v.clone() for k, v in global_model.state_dict().items()}

#     # Set up initial parameters for each client
#     client_parameters = [global_model_state.copy() for _ in range(num_clients)]

#     # Create the malicious aggregation strategy
#     malicious_aggregator = MaliciousAggregation(attack_round=attack_round, malicious_client_idx=0)

#     # Simulation loop
#     for round_num in range(100):
#         print(f"Starting federated round {round_num+1}")

#         client_updates = []

#         def train_client(client_idx):
#             client_model = CNNModel().to(device)
#             client_model.load_state_dict(client_parameters[client_idx])
#             client_train_fn = get_client_fn(client_model, train_datasets)
#             client = client_train_fn(client_idx, device=device)
#             updated_params, num_samples, metrics = client.fit()
#             return {
#                 'client_idx': client_idx,
#                 'params': updated_params,
#                 'loss': metrics['loss']
#             }

#         # Optionally parallelize client training
#         for client_idx in range(num_clients):
#             client_update = train_client(client_idx)
#             client_updates.append(client_update)

#         # Aggregate updates using the modified aggregation function
#         q = 1.0  # Adjust as needed
#         aggregated_update = malicious_aggregator.aggregate(round_num, client_updates, q)
#         global_model.load_state_dict(aggregated_update)

#         # Update client parameters
#         client_parameters = [aggregated_update.copy() for _ in range(num_clients)]

#         print(f"Round {round_num+1} completed.")

#     # Save the final model
#     torch.save(global_model.state_dict(), f"global_model_{num_clients}_clients.pth")



# main.py
# import torch
# from torch.utils.data import DataLoader
# from client import get_client_fn
# from evaluate import get_evaluate_fn
# from models import CNNModel
# from datasets import get_mnist
# from attack import MaliciousAggregation

# def main(num_clients, attack_round):
#     SEED = 0
#     torch.manual_seed(SEED)
    
#     # Load MNIST dataset
#     train, test = get_mnist()
    
#     # Split dataset among clients
#     num_train_samples = len(train)
#     client_dataset_size = num_train_samples // num_clients
#     remainder = num_train_samples % num_clients
    
#     train_datasets = []
#     start_idx = 0
#     for i in range(num_clients):
#         end_idx = start_idx + client_dataset_size + (1 if i < remainder else 0)
#         train_subset = torch.utils.data.Subset(train, range(start_idx, end_idx))
#         train_datasets.append(train_subset)
#         start_idx = end_idx
    
#     test_loaders = [("all", DataLoader(test, batch_size=64, num_workers=4))]
    
#     # Initialize the global model
#     global_model = CNNModel()
#     global_model_state = {k: v.clone() for k, v in global_model.state_dict().items()}
    
#     # Set up initial parameters for each client
#     client_parameters = [global_model_state.copy() for _ in range(num_clients)]
    
#     # Create the malicious aggregation strategy
#     malicious_aggregator = MaliciousAggregation(attack_round=attack_round)
    
#     # Simulation loop
#     for round_num in range(10):
#         print("Starting federated round " + str(round_num+1))
        
#         client_updates = []
#         for client_idx in range(num_clients):
#             # Initialize client model and load parameters
#             client_model = CNNModel()
#             client_model.load_state_dict(client_parameters[client_idx])
#             print("started get_cl func")
#             client_train_fn = get_client_fn(client_model, train_datasets)
#             print("ended get_cl func")
#             print("started cl_tr func")
#             client = client_train_fn(client_idx)
#             print("ended cl_tr func")
#             print("started .fit func")
#             updated_params, num_samples, metrics = client.fit()
#             print("ended .fit func")
#             client_updates.append({'client_idx': client_idx, 'params': updated_params})
        
#         # with concurrent.futures.ThreadPoolExecutor() as executor:
#         #     futures = [executor.submit(train_client, idx) for idx in range(num_clients)]
#         #     for future in concurrent.futures.as_completed(futures):
#         #         client_update = future.result()
#         #         client_updates.append(client_update)
        
#         print("started aggregation")
#         # Aggregate updates
#         aggregated_update = malicious_aggregator.aggregate(round_num, client_updates)
#         global_model.load_state_dict(aggregated_update)
#         print("ended aggregation")
        
#         # Update client parameters
#         client_parameters = [aggregated_update.copy() for _ in range(num_clients)]
        
#         print("Completed federated round " + str(round_num+1))
    
#     # Save the final model
#     # torch.save(global_model.state_dict(), f"global_model_{num_clients}_clients.pth")

# if __name__ == "__main__":
#     for attack_round in [0]:
#         for num_clients in [3]:
#             main(num_clients, attack_round)




# import torch
# from torch.utils.data import DataLoader
# from client import get_client_fn
# from evaluate import get_evaluate_fn
# from models import ResNet50
# from datasets import get_cifar10, ClassSubsetDataset
# from attack import MaliciousAggregation  # Assuming this handles malicious aggregation logic


# def ndarrays_to_tensors(self, ndarrays, parameters):
#         tensors = {}
#         for i, (name, param) in enumerate(parameters.items()):
#             tensors[name] = torch.tensor(ndarrays[i], dtype=param.dtype, device=param.device)
#         return tensors

# def main(num_clients, attack_round):
#     SEED = 0
#     torch.manual_seed(SEED)

#     # Load CIFAR-10 dataset and split among clients
#     train, test = get_cifar10()

#     # Calculate the size for each client dataset, and handle remaining samples
#     num_train_samples = len(train)
#     client_dataset_size = num_train_samples // num_clients
#     remainder = num_train_samples % num_clients

#     # Create a list for the datasets, distributing the remainder among clients
#     train_datasets = []
#     for i in range(num_clients):
#         start_idx = i * client_dataset_size
#         end_idx = start_idx + client_dataset_size + (1 if i < remainder else 0)
#         train_subset = torch.utils.data.Subset(train, range(start_idx, end_idx))
#         train_datasets.append(ClassSubsetDataset(train_subset))  # Assuming this accepts a subset directly

#     # Create a list of test datasets
#     tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

#     # Create dataloaders for each client
#     train_loaders = [DataLoader(t, batch_size=32, shuffle=True, num_workers=1) for t in train_datasets]
#     test_loaders = [(s, DataLoader(c, batch_size=32, num_workers=1)) for s, c in tests]

#     # Initialize the global model and its state
#     global_model = ResNet50()
#     global_model_state = {k: v.clone() for k, v in global_model.state_dict().items()}

#     # Set up initial parameters for each client
#     client_parameters = []
#     for client_idx in range(num_clients):
#         client_parameters.append(global_model_state.copy())  # each client starts with the global model's parameters

#     # Create the malicious aggregation strategy
#     malicious_aggregator = MaliciousAggregation(attack_round=attack_round)

#     # Define evaluation function (adapted from Flower setup)
#     def evaluate_global_model(model, test_loaders):
#         evaluate_fn = get_evaluate_fn(model, test_loaders)
#         return evaluate_fn()

#     # Simulate federated rounds
#     for round_num in range(100):  # Assuming 100 rounds
#         print(f"Starting federated round {round_num+1}")

#         # Each client trains independently and returns updated parameters
#         client_updates = []
#         for client_idx, train_loader in enumerate(train_loaders):
#             client_model = ResNet50().to("cpu")
#             client_model.load_state_dict(client_parameters[client_idx])
#             client_train_fn = get_client_fn(client_model, train_loaders)
#             updated_model = client_train_fn(client_idx)  # Ensure this is returning an updated model

#             # Assuming the updated_model is returned properly and you can access its state_dict
#             client_updates.append({'client_idx': client_idx, 'params': updated_model.model.state_dict()})

#         # print("client_updates: ", client_updates)
#         aggregated_update = malicious_aggregator.aggregate(round_num, client_updates)

#         # Update the global model with aggregated parameters
#         global_model.load_state_dict(aggregated_update)

#         # Optionally evaluate the global model
#         # metrics = evaluate_global_model(global_model, test_loaders)
#         print(f"Round {round_num+1} ")

#     # Save final model
#     torch.save(global_model.state_dict(), f"global_model_{num_clients}_clients.pth")

# if __name__ == "__main__":
#     # Run the simulation with different attack rounds and client counts
#     for attack_round in [0, 80]:
#         for num_clients in [3, 10, 30]:
#             main(num_clients, attack_round)




# import torch
# from torch.utils.data import DataLoader
# from client import get_client_fn
# from evaluate import get_evaluate_fn
# from models import ResNet50
# from datasets import get_cifar10, ClassSubsetDataset
# from attack import MaliciousAggregation  # Assuming this handles malicious aggregation logic



# def main(num_clients, attack_round):
#     SEED = 0
#     torch.manual_seed(SEED)

#     # Load CIFAR-10 dataset and split among clients
#     train, test = get_cifar10()

#     # Calculate the size for each client dataset, and handle remaining samples
#     num_train_samples = len(train)
#     client_dataset_size = num_train_samples // num_clients
#     remainder = num_train_samples % num_clients

#     # Create a list for the datasets, distributing the remainder among clients
#     train_datasets = []
#     for i in range(num_clients):
#         start_idx = i * client_dataset_size
#         end_idx = start_idx + client_dataset_size + (1 if i < remainder else 0)
#         train_subset = torch.utils.data.Subset(train, range(start_idx, end_idx))
#         train_datasets.append(ClassSubsetDataset(train_subset))  # Assuming this accepts a subset directly

#     # Create a list of test datasets
#     tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

#     # Create dataloaders for each client
#     train_loaders = [DataLoader(t, batch_size=32, shuffle=True, num_workers=16) for t in train_datasets]
#     test_loaders = [(s, DataLoader(c, batch_size=32, num_workers=16)) for s, c in tests]

#     # Initialize the global model and its state
#     global_model = ResNet50()
#     global_model_state = {k: v.clone() for k, v in global_model.state_dict().items()}
    
#     # Set up initial parameters for each client
#     client_parameters = []
#     for client_idx in range(num_clients):
#         client_parameters.append(global_model_state.copy())  # each client starts with the global model's parameters

#     # Create the malicious aggregation strategy
#     malicious_aggregator = MaliciousAggregation(attack_round=attack_round)

#     # Define evaluation function (adapted from Flower setup)
#     def evaluate_global_model(model, test_loaders):
#         evaluate_fn = get_evaluate_fn(model, test_loaders)
#         return evaluate_fn()

#     # Simulate federated rounds
#     for round_num in range(100):  # Assuming 100 rounds
#         print(f"Starting federated round {round_num+1}")

#         # Each client trains independently and returns updated parameters
#         client_updates = []
#         for client_idx, train_loader in enumerate(train_loaders):
#             client_model = ResNet50()
#             client_model.load_state_dict(client_parameters[client_idx])
#             client_train_fn = get_client_fn(client_model, train_loader)
#             updated_model = client_train_fn(client_idx)  
#             client_updates.append(updated_model.state_dict())  

#         aggregated_update = malicious_aggregator.aggregate(round_num, client_updates)
        
#         # Update the global model with aggregated parameters
#         global_model.load_state_dict(aggregated_update)

#         # Optionally evaluate the global model
#         metrics = evaluate_global_model(global_model, test_loaders)
#         print(f"Round {round_num+1} metrics: {metrics}")

#     # Save final model
#     torch.save(global_model.state_dict(), f"global_model_{num_clients}_clients.pth")

# if __name__ == "__main__":
#     # Run the simulation with different attack rounds and client counts
#     for attack_round in [0, 80]:
#         for num_clients in [3, 10, 30]:
#             main(num_clients, attack_round)


# #import numpy as np
# import torch
# from torch.utils.data import DataLoader, random_split
# import flwr as fl
# import time

# from client import get_client_fn
# from evaluate import get_evaluate_fn
# from models import ResNet50, ResNet18
# from datasets import get_cifar10, ClassSubsetDataset
# from attack import MalStrategy

# import ray
# ray.init(ignore_reinit_error=True, log_to_driver=True)

# def main(num_clients, attack_round):

#     SEED = 0
#     #random.seed(SEED)
#     #np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     train, test = get_cifar10(subset_fraction=0.01)

#     t1 = time.time()
#     trains = [ClassSubsetDataset(train, num=len(train) // num_clients)] + random_split(train, [1 / num_clients] * num_clients)
#     tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]
#     print("class subset executed: ", time.time()-t1)
    
#     t2 = time.time()
#     # for 4 gpus
#     train_loaders = [DataLoader(t, batch_size=32, shuffle=True, num_workers=1) for t in trains]
#     test_loaders = [(s, DataLoader(c, batch_size=32, num_workers=1)) for s, c in tests]
#     print("train and test data loaded: ", time.time()-t2)

#     t3 = time.time()
#     strategy = MalStrategy(
#     name="Sindhuja",
#     attack_round=attack_round,
#     initial_parameters=fl.common.ndarrays_to_parameters(
#         [val.numpy() for n, val in ResNet18().state_dict().items() if 'num_batches_tracked' not in n]
#     ))
   

#     print("strategy loaded: ", time.time()-t3)

#     t4 = time.time()
#     metrics = fl.simulation.start_simulation(
#         client_fn=get_client_fn(ResNet18, train_loaders),
#         num_clients= 1,
#         config=fl.server.ServerConfig(num_rounds=1, round_timeout=60),
#         strategy=strategy,
#         client_resources={"num_cpus": 0.5e}
#     )
#     print("metrics loaded: ", time.time()-t4)

# if __name__ == "__main__":
#     print("running main.py....")
#     for attack_round in [0]:
#         for num_clients in [1]:
#             main(num_clients, attack_round)
