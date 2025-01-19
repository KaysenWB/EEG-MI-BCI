import numpy as np
from matplotlib import pyplot as plt
from models import EEGNet_Conv
from dataloader import BCIDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

class processor():
    def __init__(self, args):
        self.args = args
        self.batch = args.batch
        self.epochs = args.epochs
        self.save_root = args.save_dir
        self.preds = args.preds
        self.Acc = 0
        self.kappa = 0
        self.f1 = 0

        self.model = EEGNet_Conv(args)
        self.train_dataset = BCIDataset(args, training=True)
        print(f'train data: {len(self.train_dataset)}')
        self.test_dataset = BCIDataset(args,  training=False)
        print(f'test data: {len(self.test_dataset)}')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch, shuffle=True)
        metrics = np.zeros((self.epochs, 4))
        for ep in range(self.epochs):
            self.model.train()
            loss_ac = 0
            for batch_id, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                out_y = self.model(batch[0])
                loss = self.criterion(out_y, batch[1])
                loss.backward()
                self.optimizer.step()
                loss_ac += loss.item()
                if batch_id % 10 == 0:
                    print(f'train_loss: {loss}')
                    loss_ac = 0


            Acc, Kappa, F1 = self.test()
            print(f'Epochs: {ep}, Acc: {Acc:.3f}, Kappa: {Kappa:.3f}, F1: {F1:.3f} , '
                  f'Best:{self.Acc:.3f}, {self.kappa:.3f}, {self.f1:.3f}')
            metrics[ep, :] = [ep, Acc, Kappa,F1]
            if Acc > self.Acc:
                model_path = self.save_root + '/best_.tar'
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, model_path)
                self.Acc = Acc
                self.kappa = Kappa
                self.f1 = F1

        np.savetxt(self.save_root + '/metrics.csv', metrics)


    def metrics(self, labels, out_y):
        _, predicted = torch.max(out_y.data, 1)

        Acc = accuracy_score(labels, predicted)
        Kappa =cohen_kappa_score(labels, predicted)
        F1 = f1_score(labels, predicted, average='macro')

        return Acc, Kappa, F1

    def test(self, training = True):

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch, shuffle= False)
        if training == False:
            cheakpoint = torch.load(self.save_root + '/best_.tar')
            self.model.load_state_dict(cheakpoint['state_dict'])

        Acc, Kappa, F1 = [],[],[]
        self.model.eval()
        for batch_id, batch in enumerate(self.test_loader):
            out_y = self.model(batch[0])
            ac, ka, f  = self.metrics(batch[1], out_y)
            Acc.append(ac)
            Kappa.append(ka)
            F1.append(f)

        if training == False:
            print(f'TEST: Acc: {np.mean(Acc)}, Kappa: {np.mean(Kappa):.3f}, F1: {np.mean(F1):.3f} ')
            return
        else:
            return np.mean(Acc), np.mean(Kappa), np.mean(F1)

    def pred (self):

        self.pred_loader = DataLoader(self.test_dataset, batch_size=self.batch, shuffle= False)
        cheakpoint = torch.load(self.save_root + '/best_.tar')
        self.model.load_state_dict(cheakpoint['state_dict'])
        self.model.eval()
        labels, outputs = [],[]
        for batch_id, batch in enumerate(self.pred_loader):
            labels.append(batch[1])
            outputs.append(self.model(batch[0]))
        labels = torch.concat(labels, dim=0)
        outputs = torch.concat(outputs,dim=0)
        ac, ka, f = self.metrics(labels, outputs)
        print(f'Pred. Acc: {ac:.3f}, Kappa: {ka:.3f}, F1: {f:.3f} ')

        _, pred = torch.max(outputs.data, 1)
        pred = pred.detach().numpy()
        labels = labels.detach().numpy()
        np.save(self.save_root + '/Pred.npy', pred)
        np.save(self.save_root + '/Real.npy', labels)
        print('save_results')

        return
