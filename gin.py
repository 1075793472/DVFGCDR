from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear,Embedding
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            #self.lin = Linear(edge_dim, 32)
            self.emb1 = Embedding(4 + 1, 32)
            self.emb2 = Embedding(3 + 1, 32)
            # if in_channels==65:

            if in_channels == 65+32:
                self.lin = Embedding(6, 24)
                self.lin1 = Embedding(4, 8)
                # self.lin=Embedding(6,40)
                # self.lin1 = Embedding(4, 25)
            else:
                # self.lin = Embedding(6, in_channels//2)
                # self.lin1 = Embedding(4, in_channels//2)
                self.lin = Embedding(6, 48)
                self.lin1 = Embedding(4,16)

            # if in_channels == 65 + 32:
            #  self.lin1 = Linear(edge_dim, 32)
            # else:
            #     self.lin1 = Linear(edge_dim, 64)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        # if self.lin is not None:
        #     self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        # if x_r is not None:
        #     out +=  x_r
            #out += (1) * x_r
        return self.nn(out)

    def message(self, x_i: Tensor,x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin1 is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")
        # if self.lin1 is not None:
        #     edge_attr = F.leaky_relu(self.lin1(edge_attr))
        # if self.lin is not None:
        #     edge_attr1 = self.lin1(edge_attr[:, 1])
        #     edge_attr2 = self.lin(edge_attr[:,0])
        #     #edge_attr =edge_attr1+edge_attr2
        #     edge_attr = torch.cat((edge_attr1,edge_attr2),1)

        # return (x_i - x_j )+ edge_attr
       # print(x_j.shape)
        return torch.cat((x_j, edge_attr),1)

        #return torch.cat(((x_i - x_j),edge_attr.relu()),dim=1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
class GINConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super(GINConv, self).__init__(aggr='max', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_i: Tensor,x_j: Tensor) -> Tensor:
       # print('ccc')
        return x_j-x_i

    def message_and_aggregate(self, adj_t: SparseTensor,x: OptPairTensor) -> Tensor:
        #print('ddd')
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)