# attack.py

import numpy as np
import torch

class MaliciousAggregation:
    def __init__(self, attack_round=80, malicious_client_idx=0):
        self.attack_round = attack_round
        self.malicious_client_idx = malicious_client_idx  # Index of the malicious client
        self.debug = False

    def parameters_to_ndarrays(self, parameters):
        return [param.detach().cpu().numpy() for param in parameters.values()]

    def ndarrays_to_tensors(self, ndarrays, reference_parameters):
        tensors = {}
        for i, (name, param) in enumerate(reference_parameters.items()):
            tensors[name] = torch.tensor(ndarrays[i], dtype=param.dtype)
        return tensors

    def aggregate(self, server_round, client_updates, q=1.0):
        num_clients = len(client_updates)
        epsilon = 1e-10  # Small constant to prevent division by zero

        # Collect losses and accuracies
        client_losses = [update['loss'] for update in client_updates]
        client_accuracies = [update['accuracy'] for update in client_updates]
        client_params = [update['params'] for update in client_updates]

        # Compute q-FedAvg weights based on losses
        losses = np.array(client_losses)
        weights = np.power(losses + epsilon, q)
        weights = weights / np.sum(weights)

        # Check if attack is active
        if server_round >= self.attack_round:
            print("Attack is active.")

            # Identify the malicious client
            malicious_idx = self.malicious_client_idx

            # Generate malicious update
            target_parameters = client_params[malicious_idx]
            # For simplicity, use the parameters from another client as prediction
            predicted_parameters = client_params[(malicious_idx + 1) % num_clients]

            # Convert parameters to ndarrays
            target_ndarrays = self.parameters_to_ndarrays(target_parameters)
            predicted_ndarrays = self.parameters_to_ndarrays(predicted_parameters)

            # Create malicious update
            malicious_ndarrays = [
                (t * num_clients - p * (num_clients - 1))
                for p, t in zip(predicted_ndarrays, target_ndarrays)
            ]
            # Convert back to tensors
            malicious_parameters = self.ndarrays_to_tensors(malicious_ndarrays, target_parameters)

            # Replace the malicious client's parameters with the malicious update
            client_params[malicious_idx] = malicious_parameters

            # Optionally, adjust the malicious client's loss to affect the weights
            # For example, set its loss to a high value to increase its weight
            losses[malicious_idx] = np.max(losses) * 10  # Increase the loss artificially
            weights = np.power(losses + epsilon, q)
            weights = weights / np.sum(weights)

        # Aggregate updates using q-FedAvg
        aggregated_update = {}
        for k in client_params[0].keys():
            aggregated_param = sum(
                weights[i] * client_params[i][k] for i in range(num_clients)
            )
            aggregated_update[k] = aggregated_param

        print("Aggregation done")
        return aggregated_update




# import numpy as np
# import torch
# from collections import OrderedDict
# from io import BytesIO
# from typing import cast


# class MaliciousAggregation:
#     def __init__(self, attack_round=80):
#         self.attack_round = attack_round
#         self.debug = False

#     def parameters_to_ndarrays(self, parameters):
#         return [param.detach().cpu().numpy() for param in parameters.values()]


#     def aggregate(self, server_round, client_updates):
#         # server_round: current round of training
#         # client_updates: list of (client_id, parameters) from each client

#         num_clients = len(client_updates)  # Total number of clients
        
#         # Ensure that the malicious attack is performed only after the specified attack round
#         if server_round >= self.attack_round:
#             target_parameters = client_updates[0]['params']  # Get the first client's parameters (target)

#             if self.debug:
#                 predicted_parameters = [np.mean([u[1] for u in client_updates[2:]], axis=0)]  # Debug aggregation
#             else:
#                 predicted_parameters = client_updates[1]['params']  # Use the second client's parameters as prediction

#             target_parameters = self.parameters_to_ndarrays(client_updates[1]['params'])

#             if self.debug:
#                 # Use mean aggregation for debugging
#                 predicted_parameters = [
#                     np.mean([self.parameters_to_ndarrays(update['params'])[i] for update in client_updates[2:]], axis=0)
#                     for i in range(len(target_parameters))
#                 ]
#             else:
#                 # Default predicted parameters from second client
#                 predicted_parameters = self.parameters_to_ndarrays(client_updates[1]['params'])

#             malicious_parameters = [(t * num_clients - p * (num_clients - 1)) for p, t in zip(predicted_parameters, target_parameters)]

#             aggregated_update = {
#             param_name: torch.tensor(malicious_parameters[i], dtype=param_value.dtype, device=param_value.device)
#             for i, (param_name, param_value) in enumerate(client_updates[0]['params'].items())
#             }
#         else:
#             # Normal averaging if not in attack round
#             aggregated_update = {
#                 k: sum([client['params'][k] for client in client_updates]) / num_clients
#                 for k in client_updates[0]['params'].keys()
#             }

#         print("attack done")

        # # client_updates[1] = (client_updates[1]['params'], malicious_parameters)  # Replace malicious client's update
        # np.save(f"outputs/aggregated_update_round_{server_round}.npy", 
        #     np.array([aggregated_update], dtype=object), allow_pickle=True)

        # print("malicious updates saved in a file")
        
        
        return aggregated_update

    def save_updates(self, server_round, client_updates, file_name):
        np.save(f"outputs/updates_round_{server_round}_{file_name}.npy", 
                np.array([u[1] for u in client_updates], dtype=object), allow_pickle=True)


# from functools import reduce
# import numpy as np
# import flwr as fl
# from flwr.common import (
#     FitRes,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )

# class MalStrategy(fl.server.strategy.FedAvg):  # IMPORTANT: the attack is on the client not the strategy
#     def __init__(self, name="", attack_round=80, *args, **kwargs):
#         self.debug = False
#         self.name = name
#         self.attack_round = attack_round
#         super().__init__(*args, **kwargs)

#     def aggregate_fit(self, server_round, results, failures):

#         num_clients = len(results) - 1

#         results = sorted(results, key=lambda x : x[0].cid)

#         if server_round >= self.attack_round:

#             target_parameters = parameters_to_ndarrays(results[0][1].parameters)

#             if self.debug:
#                 weights_results = [
#                     parameters_to_ndarrays(i[1].parameters) for i in results
#                 ][2:]
#                 predicted_parameters = [
#                     reduce(np.add, layer) / (num_clients - 1) for layer in zip(*weights_results)
#                 ]
#             else:
#                 predicted_parameters = parameters_to_ndarrays(results[1][1].parameters)

#             # num_clients clients: (num_clients-1) clean + 1 malicious
#             malicious_parameters = [(t * num_clients - p * (num_clients - 1)) / 1 for p,t in zip(predicted_parameters, target_parameters)]
#             results[1][1].parameters = ndarrays_to_parameters(malicious_parameters)

#         results = results[1:]

#         np.save(f"outputs/updates_round_{server_round}_{self.name}.npy", np.array([i[1] for i in results], dtype=object), allow_pickle=True)

#         return super().aggregate_fit(server_round, results, failures)