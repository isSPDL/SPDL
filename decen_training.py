import os
import shutil
import time
import argparse
import copy

import grpc
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from torch.nn.modules import loss

import Tensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import random
import matplotlib.pyplot as plt

import bc_enum
from blockchain import Blockchain
import p2p
# import matplotlib; matplotlib.use('TkAgg')


EPOCHS = 100 # number of epochs
epsilon = 1e-1
global grad_receive
# Linear Regression Model
# must inherit torch.nn.Module (nn: neural network)
# at least contains: __init__() and forward()
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # dimension of input and output is 1
    def forward(self, x):
        out = self.linear(x)
        return out


def stationay_graph(world_size):
    W = torch.zeros([world_size,world_size])
    for i in range(world_size):
        for j in range(world_size):
            W[i,j] = 1/(world_size)
    return W


def Num_Bft(world_size):
    return int((world_size-2)/2)


def coordinate(rank,world_size):
    global loss
    model = torch.tensor([0, 0])
    output = open(filename, "w")
    time_cost = 0
    for epoch in range(EPOCHS):
        print(epoch, "in total", EPOCHS)
        W = stationay_graph(world_size)  # beta
        dia = 0
        for i in range(world_size):
            if W[i, i] > 1e-3:
                dia += 1
        num_of_edge = (sum(W[W > 1e-3]) - dia) // 2
        # print(W,num_of_edge)
        t1 = time.time()
        t2 = time.time()
        time_cost += t2 - t1

        # loss = torch.FloatTensor([0])
        # torch.div(loss,world_size,rounding_mode='trunc')
        loss.div_(world_size)
        # evaluate on validation set
        # _ ,prec1 = validate(val_loader, model, criterion)
        print('loss:',loss.item())
        output.write('%d %3f\n' % (epoch, loss.item()))
        # output.write('%d %3f %3f %d\n'%(epoch,time_cost,loss.item(),num_of_edge))
        # print(epoch,time_cost,loss.item(),num_of_edge)
        output.flush()

    output.close()


def training_process(rank, world_size):
    global EPOCHS, loss
    filename = "./log/"+"loss"+str(rank)+".txt"

    log_loss = open(filename, "w")
    if rank == 1:
        data = np.loadtxt("./data/data1.txt")
    if rank == 2:
        data = np.loadtxt("./data/data2.txt")
    if rank == 3:
        data = np.loadtxt("./data/data3.txt")
    if rank == 4:
        data = np.loadtxt("./data/data4.txt")
    data[rank*1000//world_size:(rank+1)*1000//world_size-1,:]
    x = torch.tensor(data[:,0])
    y = torch.tensor(data[:,1])
    print('Start node: %d  Total: %3d'%(rank, world_size))
    current_lr =0.1
    adjust = [80, 100, 120]
    model = LinearRegression() # model is linear regression
    criterion = nn.MSELoss() # MSELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr) # lr is learning rate
    model.linear.weight.data.fill_(60)
    model.linear.bias.data.zero_()
    blockchain = Blockchain()
    grad_recv = list()
    for epoch in range(EPOCHS):
        # adaptive learning rate
        if epoch in adjust:
            current_lr = current_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        inputs = torch.tensor(x, dtype=torch.float32)
        target = torch.tensor(y, dtype=torch.float32)
        inputs = torch.unsqueeze(inputs,dim=1)
        target = torch.unsqueeze(target,dim=1)
        out = model(inputs)
        loss = criterion(out, target)
        optimizer.zero_grad()
        #grad = flatten_grad(model)
        #print('grad,before',grad)
        loss.backward()
        grad_flat = flatten_grad(model,rank)
        if rank<=Num_Bft(world_size):
            grad_flat = -grad_flat
        if rank>Num_Bft(world_size):
            grad_flat = gaussian_noise(grad_flat,float(2))
        print('my grad',grad_flat)
        grad_recv.clear()
        # grad_recv = list()
        # for i in range(world_size-1):
        #     tmp_grad = copy.deepcopy(grad_flat)
        #     grad_recv.append(tmp_grad)
        copy_grad = copy.deepcopy(grad_flat)
        grad_recv.append(copy_grad)
        # (TO DO) broadcast copy_grad
        node=p2p.Node()
        b=Tensor.serialize_torch_tensor(copy_grad)
        # grad_recv.append(b)
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGEGRAD,b)
        time.sleep(5)
        for i in p2p.grad_list:
            if i not in grad_recv:
                grad_recv.append(Tensor.deserialize_torch_tensor(i))
        p2p.grad_list.clear()
        print(rank,'grad', grad_recv)
        grad_recv.sort(key = lambda x:sum(x))
        krum_grad1 = krum(grad_recv, world_size-Num_Bft(world_size)-2,world_size-Num_Bft(world_size))
        a=Tensor.serialize_torch_tensor(krum_grad1)
        # print(model.parameters())
        blockchain.consensus_process(a)
        krum_grad=Tensor.deserialize_torch_tensor(blockchain.lastBlock.krumgrad)
        print('krum_grad', krum_grad)
        print('============== EPOCH=', epoch, '==============')
        print('============== LOSS = ', loss, '==============')
        log_loss.write('%d %3f\n' % (epoch, loss))
        log_loss.flush()
        # update gradients used in optimizer.step()
        unflatten_grad(model, krum_grad)
        optimizer.step()
        # synchronization
        if blockchain.role=="leader":
            node.send_epoch()
        else:
            while True:
                if p2p.Epoch_overing:
                    # time.sleep(1)
                    break
    model.eval()
    print('w = ', model.linear.weight,model.linear.weight.item())
    print('b = ', model.linear.bias,model.linear.bias.item())



# krum algorithm -- tolerate f<N/3 Byzantine nodes
# for details, please refer to "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NIPS'17
def cul_score(vi, vecs, num):
    vec = [sum((x-vi)*(x-vi)) for x in vecs]
    vec.sort()
    score = 0
    for i in range(num):
        score += vec[i]
    return score
def krum(vecs,num,m):
    temp = (sorted(vecs, key = lambda x:cul_score(x, vecs, num+1)))[0:m]
    #print(temp)
    return sum(temp)/m

# gaussian noise
def gaussian_noise(grad, lamda):
    laplace1 = np.random.normal(0, lamda)
    laplace2 = np.random.normal(0, lamda)
    lap = torch.tensor([laplace1,laplace2])
    #print(rank,lap)
    return grad + lap

## update gradients with flatten and unflatten functions
def flatten_all(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)

def unflatten_all(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param

def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)
def flatten_grad(model,rank):
    vec = []
    for param in model.parameters():
        vec.append(param.grad.data.view(-1))
    # return torch.cat(vargsec)
    return torch.cat(vec)

def unflatten_grad(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.grad.size())))
        param.grad.data = vec[pointer:pointer + num_param].view(param.grad.size())
        pointer += num_param

def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param

# if __name__ == '__main__':
