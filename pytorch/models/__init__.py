"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as tmodels
import importlib
from models.layers.AsyncTFCriterion import AsyncTFCriterion
from models.layers.AsyncTFBase import AsyncTFBase


def create_model(args):
    if args.arch in tmodels.__dict__:  # torchvision models
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = tmodels.__dict__[args.arch](pretrained=True)
            model = model.cuda()
        else:
            print("=> creating model '{}'".format(args.arch))
            model = tmodels.__dict__[args.arch]()
    else:  # defined as script in this directory
        model = importlib.import_module('.' + args.arch, package='models').model
        if not args.pretrained_weights == '':
            print('loading pretrained-weights from {}'.format(args.pretrained_weights))
            model.load_state_dict(torch.load(args.pretrained_weights))

    # replace last layer
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        newcls = newcls[:-1] + [AsyncTFBase(newcls[-1].in_features, args.nclass, args.nhidden).cuda()]
        model.classifier = nn.Sequential(*newcls)
    elif hasattr(model, 'fc'):
        model.fc = AsyncTFBase(model.fc.in_features, args.nclass, args.nhidden).cuda()
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = AsyncTFBase(model.AuxLogits.fc.in_features, args.nclass, args.nhidden).cuda()
    else:
        newcls = list(model.children())[:-1]
        newcls = newcls[:-1] + [AsyncTFBase(newcls[-1].in_features, args.nclass, args.nhidden).cuda()]
        model = nn.Sequential(*newcls)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = AsyncTFCriterion(args).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    return model, criterion, optimizer
