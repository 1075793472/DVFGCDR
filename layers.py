import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN_conv2(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv2, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
       #  stdv = 1. / math.sqrt(self.weight.size(1))
       #  self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # input = F.dropout(x, 0.1, self.training)
        # support = torch.mm(x, self.weight)
        # output = torch.mm(G, support)
        # return output
        x = x.matmul(self.weight)
        x = G@x
        if self.bias is not None:
            x = x + self.bias
        return x
class GCNConv1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        super(GCNConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        # self.bn = nn.BatchNorm1d(in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     nn.init.xavier_uniform_(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight):
        # x = self.bn(x)
        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out
class HGNN_conv_sp(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_sp, self).__init__()
        # self.linear_x_1 = nn.Linear(256, 256)
        # self.linear_x_2 = nn.Linear(256, 128)
        # self.linear_x_3= nn.Linear(128, 64)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x=x.cuda()
        G=G.cuda()
        x = x.matmul(self.weight.cuda())
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        # x = F.relu(x)
        # x1 = torch.relu(self.linear_x_1(x))
        # x2 = torch.relu(self.linear_x_2(x1))
        # x = torch.relu(self.linear_x_3(x2))

        return x
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.act = nn.ReLU()
        if bias:
           # self.bias = Parameter(torch.Tensor(out_ft))
            self.bias = nn.Parameter(torch.zeros(out_ft))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        #x = self.dropout(x)
       #  print(x.shape)
       #  print(self.weight.shape)
       # x = x.matmul(self.weight)
        x = torch.matmul(x, self.weight.cuda())
        x = G@x
        if self.bias is not None:
            x = x + self.bias.cuda()
        # x=self.act(x)
        return x

class HGNN_conv3(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv3, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft)).cuda()
        self.act = nn.ReLU()
        if bias:
           # self.bias = Parameter(torch.Tensor(out_ft))
            self.bias = nn.Parameter(torch.zeros(out_ft)).cuda()

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = self.dropout(x)
       #  print(x.shape)
       #  print(self.weight.shape)
        x = x.matmul(self.weight)
        x = G@x
        if self.bias is not None:
            x = x + self.bias
        #x=self.act(x)
        return x
class HGNN_conv1(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv1, self).__init__()

        self.dim_in =in_ft
        self.dim_out = out_ft
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = F.relu


    def forward(self,  feats, G):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x
class HGNN_conv_shsc(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_shsc, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.act = nn.ReLU()
        self.SHSC_layer = SHSC_layer(degree=16, alpha=0.6)
        self.W = nn.Linear(in_ft, out_ft, bias=True).cuda()
        if bias:
           # self.bias = Parameter(torch.Tensor(out_ft))
            self.bias = nn.Parameter(torch.zeros(out_ft))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        #x = self.dropout(x)
       #  print(x.shape)
       #  print(self.weight.shape)

        # x=x.cuda()
        # G=G.cuda()
        # print(x)
        # print(G)
        z=self.SHSC_layer(x, G)
        z =self.W(z)
        #z = z.matmul(self.weight.cuda())
        # if self.bias is not None:
        #     x = x + self.bias.cuda()
        # x=self.act(x)
        return z
class SHSC_layer(nn.Module):
    def __init__(self, degree, alpha, args=None):
        super(SHSC_layer, self).__init__()

        self.degree = degree
        self.alpha = alpha
        self.beta=1
    def forward(self, input, G=None, adj=None):
        ori_features = input
        emb = input
        features = input

        for i in range(self.degree):
            features = self.alpha * torch.spmm(G, features)
            emb = emb + features
        emb = emb / self.degree

        emb = self.beta*emb + (1-self.beta)* ori_features

        return emb

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (degree(K): ' \
    #            + str(self.degree) + ' alpha: ' \
    #            + str(self.alpha) + ')'