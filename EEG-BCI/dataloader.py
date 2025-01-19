import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import torch


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)

        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery817000

        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):

        # Channel default is C3

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []

        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop]
                trial = trial.reshape((1, -1))
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c)

            tt = np.concatenate(t, axis=0)
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


class BCIDataset(Dataset):
    def __init__(self, args, training=True):
        self.rate = args.test_rate
        self.label_dict = args.label_dict
        self.training = training
        self.batch = args.batch
        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

        self.x, self.y = self.load_data(args.subject, args.electrodes)
        minft = self.x.min()
        maxft = self.x.max()
        self.x = ((self.x - minft)/(maxft - minft))


    def load_data(self, subject, electrodes):
        trs,cls = [],[]
        for sub in subject:
            datasets = MotorImageryDataset(f'./data/A0{sub}T.npz')
            trials_, classes_ = datasets.get_trials_from_channels(electrodes)
            trials = np.stack(trials_, axis=1).astype('float32')
            classes = [self.label_dict[la] for la in classes_[0]]

            border = int(len(classes) * (1 - self.rate))

            if self.training == True:
                trs.append(trials[:border])
                cls += classes[:border]
            else:
                trs.append(trials[border:])
                cls += classes[border:]

        trs = np.concatenate(trs,axis=0)

        return trs, cls


    def __len__(self):
        return len(self.x) - self.batch

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


