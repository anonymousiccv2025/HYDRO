import torch
import torch.nn as nn


def mean_squared_error(x, y):
    B = x.size(0)
    x = x.view(B, -1)
    y = y.view(B, -1)
    return torch.square(x - y).sum(dim=-1).mean()


def l2_error(x, y):
    B = x.size(0)
    x = x.view(B, -1)
    y = y.view(B, -1)
    return torch.sqrt(torch.square(x - y).sum(dim=-1)).mean()


class MetricCollector(nn.Module):

    def __init__(self, compute_fn=mean_squared_error):
        """
        Helper object for collecting batched target and prediction values.
        :param compute_fn: metric to calculate the collected values over.
        """
        super(MetricCollector, self).__init__()

        self.compute_fn = compute_fn

        self.targets = None
        self.predictions = None

    def update(self, targets, predictions):
        if self.targets is not None:
            self.targets = torch.concat([self.targets, targets], dim=0)
        else:
            self.targets = targets
        if self.predictions is not None:
            self.predictions = torch.concat([self.predictions, predictions], dim=0)
        else:
            self.predictions = predictions

    def compute(self):
        return self.compute_fn(self.targets, self.predictions)
