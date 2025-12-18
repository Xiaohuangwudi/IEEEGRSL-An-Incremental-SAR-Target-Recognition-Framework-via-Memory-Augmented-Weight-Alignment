import logging
import numpy as np
from tqdm import tqdm
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.mixup import *
from utils.progressively_balance import ProgressiveSampler
from utils.faceloss import *
from utils.feature_visualization import *
from collections import Counter


EPSILON = 1e-8

init_epoch = 50
init_lr = 0.1
init_milestones = [20, 30, 40]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 50
lrate = 0.01
milestones = [20, 30, 40]
lrate_decay = 0.1
batch_size = 32
weight_decay = 2e-4
num_workers = 0
T = 2

angular_criterion = AngularPenaltySMLoss(loss_type='arcface', s=30.0, m=0.4)
angular_criterion = angular_criterion.cuda()

class WA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes-self._known_classes)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # todo : every labels/list
        train_targets = self.train_dataset.labels
        train_targets = train_targets.tolist()
        nums_per_cls = []
        for n in range(self._known_classes + 2):
            nums_per_cls.append(train_targets.count(n))
        self.sampler_generator = ProgressiveSampler(self.train_dataset, epochs, train_targets, nums_per_cls)

        # self.sampler_generator.plot_line()

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=init_lr, weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)            
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)


    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch))

        for _, epoch in enumerate(prog_bar):
            if epoch < 20:
                current_momentum = 0.5
            else:
                current_momentum = 0.999 

            if isinstance(self._network, nn.DataParallel):
                if hasattr(self._network.module.convnet, 'mem'):
                    self._network.module.convnet.mem.momentum = current_momentum
            else:
                if hasattr(self._network.convnet, 'mem'):
                    self._network.convnet.mem.momentum = current_momentum

            self._network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output_dict = self._network(inputs)

                logits = output_dict['logits']
                # loss_cls = F.cross_entropy(logits, targets.long())

                features = output_dict['features']
                features = features.cuda()

                if isinstance(self._network, nn.DataParallel):
                    fc_weight = self._network.module.fc.weight
                else:
                    fc_weight = self._network.fc.weight

                features_norm = F.normalize(features, p=2, dim=1)
                weights_norm = F.normalize(fc_weight, p=2, dim=1)

                cosine_logits = F.linear(features_norm, weights_norm)
                loss_arc = angular_criterion(cosine_logits, targets.long())


                mem_features = output_dict['mem_features'] 
                att_weight = output_dict['att_weight']
                features = output_dict['features']     

                epsilon = 1e-12

                entropy = -att_weight * torch.log(att_weight + epsilon)
                loss_mem_ent = torch.mean(torch.sum(entropy, dim=1))

                loss_mem_rec = F.mse_loss(mem_features, features)
                loss = loss_arc + loss_mem_ent + loss_mem_rec
                # print("loss_cls: {}, loss_mem_ent: {}, loss_mem_rec: {}".format(loss_cls.item(), loss_mem_ent.item(), loss_mem_rec.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            if epoch%5==0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))

        if isinstance(self._network, nn.DataParallel):
            if hasattr(self._network.module.convnet, 'mem'):
                self._network.module.convnet.mem.momentum = 0.85
        else:
            if hasattr(self._network.convnet, 'mem'):
                self._network.convnet.mem.momentum = 0.85
            
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0

            # todo :sample
            sampler, _ = self.sampler_generator(epoch)
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                           sampler=sampler)

            labels = []
            for data in train_loader:
                _, _, label = data
                label = label.squeeze()
                labels.extend(label.tolist())

            print("Epoch:{}, Counter:{}".format(epoch, Counter(labels)))

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output_dict = self._network(inputs)

                logits = output_dict['logits']
                loss_clf = F.cross_entropy(logits, targets.long())

                loss_kd = _KD_loss(logits[:, :self._known_classes], self._old_network(inputs)["logits"], T)

                features = output_dict['features']
                features = features.cuda()

                if isinstance(self._network, nn.DataParallel):
                    fc_weight = self._network.module.fc.weight
                else:
                    fc_weight = self._network.fc.weight

                features_norm = F.normalize(features, p=2, dim=1)
                weights_norm = F.normalize(fc_weight, p=2, dim=1)

                cosine_logits = F.linear(features_norm, weights_norm)
                loss_arc = angular_criterion(cosine_logits, targets.long())

                mem_features = output_dict['mem_features'] 
                att_weight = output_dict['att_weight']     

                epsilon = 1e-12

                entropy = -att_weight * torch.log(att_weight + epsilon)
                loss_mem_ent = torch.mean(torch.sum(entropy, dim=1))
    
                loss_mem_rec = F.mse_loss(mem_features, features)

                loss = loss_clf + loss_kd + 0.5 * loss_arc + loss_mem_rec + loss_mem_ent
                # print("loss_clf: {}, loss_kd: {}, loss_arc: {}, loss_mem_rec: {}, loss_mem_ent: {}".format(loss_clf.item(), loss_kd.item(), loss_arc.item(), loss_mem_rec.item(), loss_mem_ent.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)

            # if epoch % 50 == 0:
            #     os.makedirs(os.path.join('E:\incremental learning\PyCIL-master\weight', str(self._cur_task)))
            #     torch.save(self._network.state_dict(), os.path.join('E:\incremental learning\PyCIL-master\weight',
            #                                                         str(self._cur_task),
            #                                                         "model.pkl"))

            prog_bar.set_description(info)
        logging.info(info)



def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]