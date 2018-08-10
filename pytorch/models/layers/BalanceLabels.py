from torch.autograd import Function, Variable
import torch.nn as nn
import torch


def populate(dict, ind, val=0):
    if ind not in dict:
        dict[ind] = val


class ScaleGrad(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        return inputs.clone()

    @staticmethod
    def backward(ctx, grad_output):
        _, weights = ctx.saved_variables
        return grad_output * weights, None


class BalanceLabels(nn.Module):
    def __init__(self):
        super(BalanceLabels, self).__init__()
        self.zerocounts = {}
        self.counts = {}
        self.total = 0

    def update_counts(self, target):
        for tt in target:
            for j, t in enumerate(tt):
                if (t == 0).all():
                    populate(self.zerocounts, j)
                    self.zerocounts[j] += 1
                else:
                    populate(self.counts, j)
                    self.counts[j] += 1
            self.total += 1

    def get_weights(self, target):
        weights = torch.zeros(*target.shape)
        for i, tt in enumerate(target):
            for j, t in enumerate(tt):
                avg = self.total / 2
                if (t == 0).all():
                    weights[i, j] = avg / float(self.zerocounts[j])
                else:
                    weights[i, j] = avg / float(self.counts[j])
        return Variable(weights)

    def forward(self, inputs, target):
        self.update_counts(target)
        weights = self.get_weights(target)
        return ScaleGrad.apply(inputs, weights.cuda())
