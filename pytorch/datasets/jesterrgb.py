""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets.charadesrgb import default_loader, cache
import numpy as np
from glob import glob
import csv


def parse_jester_labels(filename):
    labels = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            labels[line.strip()] = i
    return labels


def parse_jester_csv(filename, cls2int):
    labels = {}
    with open(filename) as f:
        for row in f:
            row = row.strip()
            vid, label = row.split(';')
            labelnumber = cls2int[label]
            labels[vid] = {'class': labelnumber}
    return labels


class Jester(data.Dataset):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None):
        self.num_classes = 27
        self.transform = transform
        self.target_transform = target_transform
        self.cls2int = parse_jester_labels('/nfs.yoda/gsigurds/jester/jester-v1-labels.csv')
        self.labels = parse_jester_csv(labelpath, self.cls2int)
        self.root = root
        self.testGAP = 50
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(root, self.labels, split)

    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, self.testGAP
        datadir = path
        image_paths, targets, ids, times = [], [], [], []

        for i, (vid, label) in enumerate(labels.iteritems()):
            iddir = datadir + '/' + vid
            lines = glob(iddir + '/*.jpg')
            n = len(lines)
            if i % 1000 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(self.num_classes).zero_()
                #for x in label:
                target[int(label['class'])] = 1
                spacing = np.linspace(0, n - 1, testGAP)
                for loc in spacing:
                    impath = '{}/{:05d}.jpg'.format(
                        iddir, int(np.floor(loc)) + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(int(np.floor(loc)) + 1)
            else:
                for ii in range(0, n - 1, GAP):
                    target = torch.IntTensor(self.num_classes).zero_()
                    #for x in label:
                    #    if x['start'] < ii/float(FPS) < x['end']:
                    target[int(label['class'])] = 1
                    impath = '{}/{:05d}.jpg'.format(
                        iddir, ii + 1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
                    times.append(ii)
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'times': times}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data['image_paths'][index]
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        meta['time'] = self.data['times'][index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, meta

    def __len__(self):
        return len(self.data['image_paths'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get(args):
    """ Entry point. Call this function to get all Something dataloaders """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_file = args.train_file
    val_file = args.val_file
    train_dataset = Jester(
        args.data, 'train', train_file, args.cache,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.inputsize),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # missing PCA lighting jitter
            normalize,
        ]))
    val_dataset = Jester(
        args.data, 'val', val_file, args.cache,
        transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            normalize,
        ]))
    valvideo_dataset = Jester(
        args.data, 'val_video', val_file, args.cache,
        transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_dataset, valvideo_dataset
