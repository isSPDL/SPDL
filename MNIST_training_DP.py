import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
import pdb
import datasets
import pickle
import bc_enum
import p2p
epsilon = 0.4
sigama=1e-5


class SoftmaxModel(nn.Module):
    def __init__(self, D_in, D_out):
        super(SoftmaxModel, self).__init__()
        self.linear = nn.Linear(D_in, D_out)
        self.D_in = D_in
        self.D_out = D_out

    def forward(self, x):
        # TODO: fix hardcoded
        x = np.reshape(x, (x.shape[0], self.D_in))
        return self.linear(x)

    # Unflattens flattened gradient
    def reshape(self, flat_gradient):
        layers = []
        layers.append(
            torch.from_numpy(np.reshape(flat_gradient[0:self.D_in * self.D_out], (self.D_out, self.D_in))).type(
                torch.FloatTensor))
        layers.append(torch.from_numpy(flat_gradient[self.D_in * self.D_out:self.D_in * self.D_out + self.D_out]).type(
            torch.FloatTensor))
        return layers


class Client():
    def __init__(self, dataset, filename, batch_size, model,port,train_cut=.80):
        # initializes dataset
        self.batch_size = batch_size
        self.port=port
        Dataset = datasets.get_dataset(dataset)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        self.trainset = Dataset(filename, "./" + dataset, is_train=True, transform=transform)
        self.testset = Dataset("mnist_test", "./" + dataset, is_train=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)
        # self.attackset = Dataset("mnist_digit1", "../mnist_data/" + dataset, is_train=False, transform=transform)
        # self.attackloader = torch.utils.data.DataLoader(self.attackset, batch_size=len(self.testset), shuffle=False)
        self.model = model

        ### Tunables ###
        # self.criterion = nn.MultiLabelMarginLoss()
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.75, weight_decay=0.001)  # mnist_cnn
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.001) # mnist_softmax
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_cnn
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_softmax
        self.aggregatedGradients = []
        self.loss = 0.0

    # TODO:: Get noise for diff priv
    def getGrad(self):
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()


            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 100)
            self.loss = loss.item()

            # TODO: Find more efficient way to flatten params
            # get gradients into layers
            layers = np.zeros(0)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    layers = np.concatenate((layers, param.grad.numpy().flatten()), axis=None)
            return layers

    # Called when an aggregator receives a new gradient
    def updateGrad(self, gradient):
        # Reshape into original tensor
        layers = self.model.reshape(gradient)
        self.aggregatedGradients.append(layers)

    # Step in the direction of provided gradient.
    # Used in BlockML when gradient is aggregated in Go
    def simpleStep(self, gradient):
        print("Simple step")
        layers = self.model.reshape(gradient)
        # Manually updates parameter gradients
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = layers[layer]
                layer += 1

                # Step in direction of parameter gradients
        self.optimizer.step()

    # Called when sufficient gradients are aggregated to generate updated model
    def step(self):
        # Aggregate gradients together in place
        for i in range(1, len(self.aggregatedGradients)):
            gradients = self.aggregatedGradients[i]
            for g, gradient in enumerate(gradients):
                self.aggregatedGradients[0][g] += gradient

        # Average gradients
        for g, gradient in enumerate(self.aggregatedGradients[0]):
            gradient /= len(self.aggregatedGradients)

        # Manually updates parameter gradients
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.aggregatedGradients[0][layer]
                layer += 1

        # Step in direction of parameter gradients
        self.optimizer.step()
        self.aggregatedGradients = []

    # Called when the aggregator shares the updated model
    def updateModel(self, modelWeights):

        layers = self.model.reshape(modelWeights)
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = layers[layer]
                layer += 1

    def getModelWeights(self):
        layers = np.zeros(0)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layers = np.concatenate((layers, param.data.numpy().flatten()), axis=None)
        return layers

    def getLoss(self):
        return self.loss

    def getModel(self):
        return self.model

    def getTestErr(self):
        for i, data in enumerate(self.testloader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()
            inputs, labels = Variable(inputs), Variable(labels)
            out = self.model(inputs)
            pred = np.argmax(out.detach().numpy(), axis=1)
        return 1 - accuracy_score(pred, labels)



def returnModel(D_in,D_out):
    model = SoftmaxModel(D_in, D_out)
    # model = MNISTCNNModel()
    return model
def gaussian_noise(grad):
    noise_grad=grad
    global epsilon,sigama
    sum1=0
    for i in range(noise_grad.size):
        sum1=sum1+noise_grad[i]*noise_grad[i]
    a=math.sqrt(sum1)
    for i in range(noise_grad.size):
        gauss1 = np.random.normal(0,(1e-6)*a*math.sqrt(2*math.log(1.25/sigama))/epsilon)
        noise_grad[i]+=gauss1
    return noise_grad
def BFT(grad):
    bft_grad=grad
    for i in range(bft_grad.size):
        bft_grad[i]=10000000
    return bft_grad
def Num_Bft(world_size,f):
    return int(world_size*f)
def run(f):
    iter_time = 100
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    batch_size = 100
    train_cut = 0.8
    node_size=4
    num_bft = Num_Bft(node_size, f)
    model = returnModel(D_in, D_out)
    client=Client("mnist", "mnist" + str(int(int(p2p.PORT)-50051)//2), batch_size, model, p2p.PORT, train_cut)
    node=p2p.Node()
    # filename1 = "./log/4/DP/loss/" + "loss_" + str(node.PORT) + ".txt"
    # filename2 = "./log/4/DP/error/" + "Test_error_" + str(node.PORT) + ".txt"
    filename3 = "./log/30/DP/time/" + "time_correspond" + str(node.PORT) + ".txt"
    filename4 = "./log/30/DP/time/" + "time_compute" + str(node.PORT) + ".txt"
    # log_loss1 = open(filename1, "w")
    # log_loss2 = open(filename2, "w")
    log_time1 = open(filename3, "w")
    log_time2 = open(filename4, "w")
    for iter in range(iter_time):
        # t1 = time.time()
        grad_recv = list()
        t1=time.time()
        if ((int(p2p.PORT) - 50051) // 2) < (node_size - num_bft):
            grad = client.getGrad()
        else:
            grad = BFT(client.getGrad())
        t2=time.time()
        grad_recv.append(grad)
        grad1=gaussian_noise(grad)
        a=pickle.dumps(grad1)
        t3=time.time()
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGEGRAD, a)
        t4=time.time()
        time.sleep(3)
        for i in p2p.grad_list:
            if i not in grad_recv:
                grad_recv.append(pickle.loads(i.para))
        p2p.grad_list.clear()
        # Share updated model
        for i in grad_recv:
            client.updateGrad(i)
        client.step()
        # t2 = time.time()
        # t=t2-t1-3
        t_cor = t4 - t3
        t_com = t2 - t1
        # print('============== time=', t, '==============')
        print('============== EPOCH=', iter, '==============')
        print('============== time_cor=', t_cor, '==============')
        print('============== time_com=', t_com, '==============')
        # print('============== LOSS = ', client.getLoss(), '==============')
        # print('============== ERROR = ', client.getTestErr(), '==============')
        # log_loss1.write('%d %3f\n' % (iter, client.getLoss()))
        # log_loss2.write('%d %3f\n' % (iter, client.getTestErr()))
        # log_loss1.flush()
        # log_loss2.flush()
        log_time1.write('%d %3f\n' % (iter, t_cor))
        log_time2.write('%d %3f\n' % (iter, t_com))
        log_time1.flush()
        log_time2.flush()