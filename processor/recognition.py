#!/usr/bin/env python
# pylint: disable=W0201
import argparse
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt

# torchlight
from torchlight import str2bool

from .processor import Processor




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,**(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, - k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        #统计
        sumNumber = np.zeros(400)
        wrongNumber = np.zeros(400)
        wrongList = np.zeros([400,400])
        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

                resultLabel = output.argmax(dim=1)      #argmax 返回指定维度最大值的序号
                for i in range(len(label)):
                    sumNumber[label[i]] = sumNumber[label[i]] + 1
                    if label[i] != resultLabel[i]:
                        wrongNumber[label[i]] = wrongNumber[label[i]] + 1
                    wrongList[label[i]][resultLabel[i]] = wrongList[label[i]][resultLabel[i]] + 1

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
            '''
            # label名称:400
            label_name_path = './resource/kinetics_skeleton/label_name.txt'
            with open(label_name_path) as f:
                self.label_name = [line.rstrip() for line in f.readlines()]

            #结果排序
            with open("C:/Users/57641/Desktop/rank.txt", "w", encoding='utf-8') as f:

                totalSum = 0
                totalWrong = 0
                for i in range(len(sumNumber)):
                    totalSum = totalSum + sumNumber[i]
                    totalWrong = totalWrong + wrongNumber[i]
                print(float(totalSum)/float(totalWrong))

                f.write("Total" + str(round(float(totalWrong)/float(totalSum)*100,2))+"%"+"   Wrong/Sum:"+str(totalWrong)+"/"+str(totalSum))
                f.write("\n")

                resultList = zip(self.label_name,sumNumber,wrongNumber,wrongList)
                resultRank = sorted(resultList, key=lambda item: item[2]/item[1])
                for i in range(len(resultRank)):
                    print(i,resultRank[i][0])
                    print("   Wrong Rate", round(resultRank[i][2] / resultRank[i][1]*100, 2),"%        Wrong/Sum ",resultRank[i][2],resultRank[i][1])
                    f.write(str(i)+" "+str(resultRank[i][0]))
                    f.write("\n")
                    f.write("   Wrong Rate:"+str(round(resultRank[i][2] / resultRank[i][1]*100, 2))+"%   ,Wrong/Sum:"+str(resultRank[i][2])+"/"+str(resultRank[i][1]))
                    f.write("\n")
                    wrongRank = np.argsort(-resultRank[i][3])
                    for j in range(6):
                        f.write("       "+str(j)+": "+str(self.label_name[wrongRank[j]])+"   Number:"+ str(resultRank[i][3][wrongRank[j]])
                                +"   Rate:"+ str(round(resultRank[i][3][wrongRank[j]]/resultRank[i][1]*100,2))+"%")
                        f.write("\n")
                        #print("   ",j,self.label_name[wrongRank[j]],resultRank[i][3][wrongRank[j]], float(resultRank[i][3][wrongRank[j]])/resultRank[i][2])

                f.close()
                '''
            '''      
            percent = np.zeros_like(sumNumber)
            for i in range(len(sumNumber)):
                percent[i] = float(wrongNumber[i]) / float(sumNumber[i]) * 100.0

            plt.barh(range(len(percent)), percent, height=0.7, color='steelblue', alpha=0.8)
            plt.yticks(range(len(percent)), self.label_name)

            plt.title("识别错误占比")
            plt.xlabel("类别")
            plt.ylabel("数量")
            plt.legend()
            plt.show()
            '''

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
