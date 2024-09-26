from .data import Data
from .sampler import BatchSampler
import torch
import numpy as np

class FlexibleCartesianProd(Data):
    """Flexible Cartesian Product input data format for MIONet architecture.

    This dataset can be used with the network ``MIONetCartesianProd`` for operator
    learning, supporting multiple branch and trunk inputs.

    Args:
        X_train: A tuple of NumPy arrays. The shape of each array depends on whether it
            is a branch or a trunk input.
        y_train: A NumPy array of shape (`N1`, `N2`).
        input_types: A list of strings indicating the type of each input in X_train/X_test.
                     Should be "branch" or "trunk".
        trunk_points: Optional number of points to sample for the trunk inputs.
        standardize: Whether to standardize branch inputs.
    """

    def __init__(self, 
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test, 
                 input_types,
                 trunk_points=None,
                 standardize=False):
        
        if len(X_train) != len(input_types) or len(X_test) != len(input_types):
            raise ValueError("Length of X_train and X_test must match the length of input_types.")

        self.train_x, self.train_y = list(X_train), y_train
        self.test_x, self.test_y = list(X_test), y_test
        self.input_types = input_types
        self.trunk_points = trunk_points
        self.scaler = []  # List to store scalers for branch inputs

        # Standardize branch inputs if required
        if standardize:
            for i, input_type in enumerate(input_types):
                if input_type == "branch" and len(self.train_x[i].shape) <= 2:
                    scaler, self.train_x[i], self.test_x[i] = utils.standardize(
                        self.train_x[i], self.test_x[i]
                    )
                    self.scaler.append(scaler)
                else:
                    self.scaler.append(None)  # No scaler for trunk inputs

        # Convert inputs to torch tensors
        dtype = torch.float32
        self.train_x = tuple(torch.tensor(x, dtype=dtype) for x in self.train_x)
        self.test_x = tuple(torch.tensor(x, dtype=dtype) for x in self.test_x)

        # Initialize samplers
        self.branch_samplers = [BatchSampler(len(x), shuffle=True) for x, t in zip(self.train_x, input_types) if t == "branch"]
        self.trunk_sampler = BatchSampler(len(next((x for x, t in zip(self.train_x, input_types) if t == "trunk"), None)), shuffle=True) if any(t == "trunk" for t in input_types) else None

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            if self.trunk_points is None:
                return self.train_x, self.train_y
            else:
                trunk_indices = self._get_trunk_indices(self.train_x)
                trunk_inputs = tuple([x[trunk_indices] if t == "trunk" else x for x, t in zip(self.train_x, self.input_types)])
                return ((trunk_inputs, self.train_y[:, trunk_indices]))
        
        if not isinstance(batch_size, (tuple, list)):
            branch_indices = self.branch_samplers[0].get_next(batch_size)
            if self.trunk_points is not None:
                trunk_indices = self._get_trunk_indices(self.train_x)
                trunk_inputs = tuple([x[trunk_indices] if t == "trunk" else x[branch_indices] for x, t in zip(self.train_x, self.input_types)])
                return trunk_inputs, self.train_y[branch_indices][:, trunk_indices]
            else:
                trunk_inputs = tuple([x if t == "trunk" else x[branch_indices] for x, t in zip(self.train_x, self.input_types)])
                return trunk_inputs, self.train_y[branch_indices]

    def test(self):
        if self.trunk_points is None:
            return self.test_x, self.test_y
        else:
            trunk_indices = self._get_trunk_indices(self.test_x)
            trunk_inputs = tuple([x[trunk_indices] if t == "trunk" else x for x, t in zip(self.test_x, self.input_types)])
            return trunk_inputs, self.test_y[:, trunk_indices]

    def _get_trunk_indices(self, x_list):
        trunk_tensor = next(x for x, t in zip(x_list, self.input_types) if t == "trunk")
        random_indices = np.random.randint(0, len(trunk_tensor), size=int(self.trunk_points))
        return random_indices
    
    def transform_inputs(self, x):
        # Separate the inputs according to their type
        x_branch = [x[i] for i, t in enumerate(self.input_types) if t == "branch"]
        x_trunk = [x[i] for i, t in enumerate(self.input_types) if t == "trunk"]

        # Scale branch inputs if scalers are available
        x_branch_scaled = []
        for i, branch_data in enumerate(x_branch):
            if i < len(self.scaler) and self.scaler[i] is not None:  # Ensure there is a scaler for this input
                x_branch_scaled.append(self.scaler[i].transform(branch_data))
            else:
                x_branch_scaled.append(branch_data)

        # Scale trunk inputs if scalers are available
        x_trunk_scaled = []
        for i, trunk_data in enumerate(x_trunk):
            if len(self.scaler) > len(x_branch) and self.scaler[len(x_branch)] is not None:
                x_trunk_scaled.append(self.scaler[len(x_branch)].transform(trunk_data))
            else:
                x_trunk_scaled.append(trunk_data)

        return tuple(x_branch_scaled + x_trunk_scaled)