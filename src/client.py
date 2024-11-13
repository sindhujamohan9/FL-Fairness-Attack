# client.py
import torch
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PyTorchClient:
    def __init__(self, cid, model, train_loader, device="cpu"):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def get_parameters(self):
        return self.model.state_dict()

    def fit(self, epochs=100, lr=0.1):
        """Trains the model for a given number of epochs and returns updated parameters and metrics."""
        optimiser = SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimiser.zero_grad()
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                optimiser.step()

                # Accumulate loss
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Compute predictions and accumulate correct predictions
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == y).sum().item()

        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        # Print the metrics
        print(f"Client {self.cid} - Epochs: {epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Return updated parameters, number of samples, and metrics
        return self.get_parameters(), total_samples, {"loss": avg_loss, "accuracy": accuracy}


    # def fit(self, epochs=2, lr=0.1):
    #     optimiser = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    #     self.model.train()
    #     total_loss = 0
    #     total_samples = 0
    #     for epoch in range(epochs):
    #         for x, y in self.train_loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             optimiser.zero_grad()
    #             z = self.model(x)
    #             loss = F.cross_entropy(z, y)
    #             loss.backward()
    #             optimiser.step()
    #             total_loss += loss.item() * x.size(0)  # Accumulate total loss
    #             total_samples += x.size(0)
    #     avg_loss = total_loss / total_samples
    #     return self.get_parameters(), len(self.train_loader.dataset), {"loss": avg_loss}

    # def fit(self, epochs=2, lr=0.01):
    #     optimiser = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    #     self.model.train()
    #     total_loss = 0
    #     for epoch in range(epochs):
    #         for x, y in self.train_loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             optimiser.zero_grad()
    #             z = self.model(x)
    #             loss = F.cross_entropy(z, y)
    #             loss.backward()
    #             optimiser.step()
    #             total_loss += loss.item()
    #     return self.get_parameters(), len(self.train_loader.dataset), {"loss": total_loss / len(self.train_loader)}

def get_client_fn(model, train_datasets):
    def client_fn(cid):
        train_loader = DataLoader(train_datasets[int(cid)], batch_size=64, shuffle=True, num_workers=4)
        return PyTorchClient(int(cid), model, train_loader)
    return client_fn


# import torch
# from torch.optim import SGD
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# class PyTorchClient:
#     def __init__(self, cid, model, train_loader, device="cpu"):
#         self.cid = cid
#         self.model = model
#         self.train_loader = train_loader
#         self.device = device

#     def set_parameters(self, parameters):
#         """Sets the model's parameters from a list of parameters."""
#         state_dict = self.model.state_dict()
#         param_names = list(state_dict.keys())
#         param_dict = dict(zip(param_names, parameters))  # Create a mapping from param names to values
#         state_dict.update(param_dict)
#         self.model.load_state_dict(state_dict)

#     def get_parameters(self):
#         """Returns the model's parameters as a list of tensors."""
#         return [val.cpu().numpy() for val in self.model.parameters()]

#     def fit(self, epochs=2, lr=0.1):
#         """Trains the model for a given number of epochs."""
#         optimiser = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        
#         self.model.train()

#         total_loss = 0
#         for epoch in range(epochs):
#             for x, y in self.train_loader:
#                 x, y = x.to(self.device), y.to(self.device)

#                 optimiser.zero_grad()

#                 z = self.model(x)
#                 loss = F.cross_entropy(z, y)

#                 loss.backward()
#                 optimiser.step()

#                 with torch.no_grad():
#                     total_loss += loss.item()

#         return self.get_parameters(), len(self.train_loader), {"loss": total_loss / epochs}

#     def evaluate(self):
#         """Evaluates the model on its current test set (optional)."""
#         self.model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for x, y in self.train_loader:
#                 x, y = x.to(self.device), y.to(self.device)
#                 outputs = self.model(x)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += y.size(0)
#                 correct += (predicted == y).sum().item()

#         accuracy = 100 * correct / total
#         return accuracy

# def get_client_fn(model, train_loaders):
#     """Returns a client function that creates a new client for each training round."""
    
#     def client_fn(cid):
#         model_instance = model.to("cpu")  # Initialize the model for this client
#         train_loader = train_loaders[int(cid)]  # Select the correct data loader
#         return PyTorchClient(int(cid), model_instance, train_loader)

#     return client_fn



# from collections import OrderedDict
# import torch
# from torch.optim import SGD
# import flwr as fl
# import torch.nn.functional as F


# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, cid, model, train_loader, device="cuda"):
#         self.cid = cid
#         self.model = model
#         self.train_loader = train_loader
#         self.device = device

#     def set_parameters(self, parameters):
#         keys = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]  # this is necessary due to batch norm.
#         params_dict = zip(keys, parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

#     def get_parameters(self, *args, **kwargs):
#         return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'num_batches_tracked' not in name]

#     def fit(self, parameters, config, epochs=2):

#         self.set_parameters(parameters)
#         # params based on: https://github.com/meliketoy/wide-resnet.pytorch
#         optimiser = SGD(self.model.parameters(), lr=self.get_lr(config["round"]), momentum=0.9, weight_decay=5e-4, nesterov=True)
        
#         self.model.train()

#         total_loss = 0
#         for epoch in range(epochs):

#             for x, y in self.train_loader:
#                 x, y = x.to(self.device), y.to(self.device)

#                 optimiser.zero_grad()

#                 z = self.model(x)
#                 loss = F.cross_entropy(z, y)

#                 loss.backward()
#                 optimiser.step()

#                 with torch.no_grad():
#                     total_loss += loss

#         return self.get_parameters(), len(self.train_loader), {"loss": total_loss/epochs}

#     def evaluate(self, parameters, config):
#         return 0., len(self.train_loader), {"accuracy": 0.}

#     def get_lr(self, training_round):
#         if training_round <= 60:
#             return 0.1
#         if training_round <= 120:
#             return 0.02
#         if training_round <= 160:
#             return 0.004
#         return 0.0008

# def get_client_fn(model, train_loaders):
    
#     def client_fn(cid):
#         nonlocal model, train_loaders
#         model = model().to("cuda")  # probs should be in fit but easier here
#         train_loader = train_loaders[int(cid)]
#         return FlowerClient(int(cid), model, train_loader)

#     return client_fn