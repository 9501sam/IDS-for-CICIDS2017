#!/usr/bin/env python
# coding: utf-8

# In[8]:


from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from IPython.display import display, HTML
display(HTML("<style>.container { width:60% !important; }</style>"))


# In[9]:


class DNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.hidden1 = nn.Linear(num_inputs, 500)
        self.hidden2 = nn.Linear(500, 500)
        self.hidden3 = nn.Linear(500, 500)
        self.hidden4 = nn.Linear(500, 500)
        self.hidden5 = nn.Linear(500, 500)
        self.hidden6 = nn.Linear(500, 500)
        self.output = nn.Linear(500, num_outputs)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = self.output(x)
        return x


# In[10]:


model = DNN(83, 5)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# In[11]:


weights_values = []
for param in model.parameters():
    weights_values.append(param.data.numpy())
initial_parameters = fl.common.ndarrays_to_parameters(weights_values)


# In[12]:


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print(examples)

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
strategy = fl.server.strategy.FedAvg(initial_parameters = initial_parameters, evaluate_metrics_aggregation_fn=weighted_average, 
    min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2)


# In[ ]:


fl.server.start_server(
    server_address="127.0.0.1:53388",
    # server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)

