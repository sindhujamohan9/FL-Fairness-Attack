import torch
import torch.nn.functional as F
import numpy as np

def get_evaluate_fn(model, loaders, file_name="", device="cpu"):
    # Prepare the model and move it to the device (GPU or CPU)
    model = model(loaders).to(device)

    def evaluate(model, loaders):
        model.eval()

        overall_loss = None
        metrics = {}

        with torch.no_grad():
            # Loop through the loaders (train/test)
            for (name, loader) in loaders:
                loss = total = correct = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    # Forward pass
                    z = model(x)

                    # Calculate loss
                    loss += F.cross_entropy(z, y)

                    # Metrics calculation
                    total += y.size(0)
                    correct += (torch.max(z.data, 1)[1] == y).sum().item()

                metrics[f"loss_{name}"] = loss.item()
                metrics[f"accuracy_{name}"] = correct / total

                if name == "all":
                    overall_loss = loss / len(loader)

        # Optionally save the metrics to a file
        np.save(f"outputs/metrics_{file_name}.npy", np.array([metrics], dtype=object), allow_pickle=True)

        return overall_loss, metrics

    return evaluate



# import numpy as np
# from collections import OrderedDict
# import torch
# import torch.nn.functional as F

# def get_evaluate_fn(model, loaders, file_name="", device="cuda"):

#     model = model().to(device)

#     def evaluate(training_round, parameters, config):

#         nonlocal model, device, file_name

#         keys = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
#         params_dict = zip(keys, parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)

#         model.eval()

#         with torch.no_grad():

#             overall_loss = None
#             metrics = {}

#             for (name, loader) in loaders:

#                 loss = total = correct = 0
#                 for x, y in loader:
#                     x, y = x.to(device), y.to(device)

#                     z = model(x)
#                     loss += F.cross_entropy(z, y)

#                     total += y.size(0)
#                     correct += (torch.max(z.data, 1)[1] == y).sum().item()

#                 metrics[f"loss_{name}"] = loss.item()
#                 metrics[f"accuracy_{name}"] = correct / total

#                 if name == "all":
#                     overall_loss = loss / len(loader)

#         np.save(f"outputs/metrics_{training_round}_{file_name}.npy", np.array([metrics], dtype=object), allow_pickle=True)

#         return overall_loss, metrics

#     return evaluate