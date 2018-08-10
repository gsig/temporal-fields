# pylint: disable=W0221,E1101
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from random import random
from models.layers.VerboseGradients import VerboseGradients
from models.layers.BalanceLabels import BalanceLabels


def unit(x):
    # normalize tensor in log space to have unit sum for each row
    minx, _ = x.max(1)
    z = (x - minx[:, None]).exp().sum(1).log() + minx
    return x - z[:, None]


def lse(x, dim=None, keepdim=False):
    # log sum exp @alshedivat
    return (x - F.log_softmax(x)).sum(dim, keepdim=keepdim)


def sme(x, y, dim=None, keepdim=False):
    # Sum mul exp
    return (x * torch.exp(y)).sum(dim, keepdim=keepdim)


def axb(a, x, b):
    # a and b are batched vectors, X is batched matrix
    # returns a^t * X * b
    xb = torch.bmm(x, b[:, :, None])
    return (a * xb.squeeze()).sum(1)


def avg(iterator, weight=1.):
    # compounding weight
    item, w = next(iterator)
    total = item.clone() * w
    n = 1.
    for i, (item, w) in enumerate(iterator):
        w1 = 1. * weight**(i + 1)
        total += item * w1 * w
        n += w1
    return total / n


def winsmooth(mat, kernelsize=1):
    print('applying smoothing with kernelsize {}'.format(kernelsize))
    mat.detach()
    n = mat.shape[0]
    out = mat.clone()
    for m in range(n):
        a = max(0, m - kernelsize)
        b = min(n - 1, m + kernelsize)
        out[m, :] = mat[a:b + 1, :].mean(0)
    return out


def gtmat(sizes, target):
    # convert target to a matrix of zeros and ones
    out = torch.zeros(*sizes)
    for i, t in enumerate(target):
        t = t.data[0] if type(t) is torch.Tensor else t
        if len(sizes) == 3:
            out[i, t, :] = 1
        else:
            out[i, t] = 1
    if type(target) is Variable:
        return Variable(out.cuda())
    else:
        return out.cuda()


def nll_loss(soft_target, logdist, reduce=True):
    # @Hongyi_Zhang
    # assumes soft_target is normalized to 1 and between [0,1]
    # logdist is a (normalized) log distribution
    logdist = unit((logdist.exp() + 0.00001).log())  # for numerical stability
    if soft_target.dim() == 3:
        out = (-soft_target * logdist).sum(2).sum(1)
    else:
        out = (-soft_target * logdist).sum(1)
    if reduce:
        return out.mean()
    else:
        return out


class MessagePassing(object):
    # Class for keeping track of messages across frames
    def __init__(self, maxsize, w_time, decay, sigma):
        super(MessagePassing, self).__init__()
        self.maxsize = maxsize
        self.w_time = w_time
        self.decay = decay
        self.sigma = sigma
        self.storage = {}
        self.storage_gt = {}
        self.training = self.training if hasattr(self, 'training') else True
        self.nc = None

    def mget(self, idtime, size, storage, cond=lambda t, t0: True, kernel=lambda t, t0: 1):
        # get message using condition on the timestamps
        def meta(ids, t0):
            try:
                return avg(((y, kernel(t, t0)) for t, y in storage[ids]
                            if cond(t, t0)), 1. / self.decay)
            except (StopIteration, KeyError):
                return torch.zeros(size)
        out = [meta(ids, time) for ids, time in idtime]
        return Variable(torch.stack(out, 0).cuda())

    def get_msg(self, idtime, time='past', storage=None):
        storage = self.storage if storage is None else storage
        cond = lambda t, t0: t < t0 if time == 'past' else t > t0
        kernel = lambda t, t0: math.exp(-float(t - t0)**2 / (2 * self.sigma**2))
        return self.mget(idtime, self.nc, storage, cond, kernel) * self.w_time

    def get_gt_msg(self, idtime, time='past'):
        return self.get_msg(idtime, time, self.storage_gt)

    def mset(self, msg, idtime, storage):
        # keep a queue of size maxsize for each id
        # messages are stored in normal space
        # queue for each id is stored in the order in which the messages were stored
        for m, (ids, time) in sorted(zip(msg, idtime), key=lambda x: random()):
            if ids not in storage:
                storage[ids] = []
            data = m if type(m) is not torch.Tensor else m.data.cpu()
            storage[ids].append((time, data))
            if len(storage[ids]) > self.maxsize:
                del storage[ids][0]

    def set_msg(self, qa, idtime):
        self.mset(qa, idtime, self.storage)

    def set_gt_msg(self, qa, target, idtime):
        x = target.data.cpu()
        self.mset(x, idtime, self.storage_gt)


class AsyncTFCriterion(nn.Module, MessagePassing):
    def __init__(self, args):
        MessagePassing.__init__(self, args.memory_size, args.temporal_weight, args.memory_decay, args.sigma)
        nn.Module.__init__(self)
        self.msg_n = 5
        self.w_tloss = args.temporalloss_weight
        self.orig_loss = args.originalloss_weight
        self.adjustment = args.adjustment
        self.loss = nn.BCELoss()
        self.balanceloss = args.balanceloss
        self.BalanceLabels = BalanceLabels()
        self.winsmooth = args.window_smooth

    def forward(self, a, aa, target, id_time, n=1, synchronous=False):
        if target.dim() == 1:
            print('converting Nx1 target to NxC')
            target = Variable(gtmat(a.shape, target.data.long()))
        target = target.float()
        idtime = zip(id_time['id'], id_time['time'])
        self.nc = a.shape[1]
        a, aa = VerboseGradients.apply(a, aa)
        msg = self.get_msg(idtime, 'past')
        fmsg = self.get_msg(idtime, 'future')
        qa = a.clone()
        qa += (aa * msg[:, :, None]).sum(1)
        qa += (aa * fmsg[:, None, :]).sum(2)
        qa = torch.nn.Sigmoid()(qa)
        if self.balanceloss:
            qa = self.BalanceLabels(qa, target)
        loss = self.loss(qa, target)
        loss += self.loss(torch.nn.Sigmoid()(a), target) * self.orig_loss
        # self.set_msg(a, idtime)
        # self.set_msg(qa, idtime)
        self.set_msg(torch.nn.Sigmoid()(a), idtime)

        if self.training:
            if self.adjustment:
                # This is an adjustment that makes the objective a true fully-connected CRF
                # Can be thought of as a regularizer on aa
                print('adding asynctf adjustment to loss')
                self.set_gt_msg(qa, target, idtime)
                gt_msg = self.get_gt_msg(idtime, time='past')
                gt_fmsg = self.get_gt_msg(idtime, time='future')
                pastloss = .5 * axb(gt_msg - msg, aa, target).pow(2).mean() * self.w_tloss / self.nc**2
                futureloss = .5 * axb(target, aa, gt_fmsg - fmsg).pow(2).mean() * self.w_tloss / self.nc**2
            else:
                pastloss = loss * 0
                futureloss = loss * 0

            print('losses class: {} \t past: {} \t future: {}'.format(
                  loss.data[0], pastloss.data[0], futureloss.data[0]))
            loss = (loss + pastloss + futureloss) / 3

        if not synchronous or n > self.msg_n:
            out = qa.clone()
            if synchronous:
                out = winsmooth(out, kernelsize=self.winsmooth)
            return out, loss
        else:
            return self.forward(a, aa, target, id_time, n=n + 1, synchronous=synchronous)
