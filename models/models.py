import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
import torch.nn as nn
from torch.nn import Linear
import torch

"""
Appling pyG lib
"""

######################################################################################
# origin GAT model with 80 node features
######################################################################################

class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, k=10):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        # self.conv0 = GATConv(node_feature_dim, hidden_dim, heads=nheads)

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # edge_index, _  = dropout_adj(edge_index, p = 0.2, training = self.training)

        # x = self.conv0(x, edge_index)
        # x = self.norm0(x, batch)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.drop, training=self.training)
        # print(x.dtype)
        # print(edge_index.dtype)
        # edge_index = torch.tensor(edge_index,dtype=torch.int64)
        edge_index = edge_index.clone().detach()
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        # print(x.dtype)
        # print(edge_index.dtype)
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # 2. Readout layer
        # x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.topk_pool(x, edge_index, batch=batch)[0]
        x = x.view(batch[-1] + 1, -1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.lin0(x)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)

        z = x  # extract last layer features

        x = self.lin(x)

        return x, z


######################################################################################
# GAT model with 2048 + 80 node features
######################################################################################


class GATModel1(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, k=10):
        super(GATModel1, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        # self.conv0 = GATConv(node_feature_dim 2108, hidden_dim:64 2048 - 512 - 128, heads=nheads 4)

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # edge_index, _  = dropout_adj(edge_index, p = 0.2, training = self.training)

        # x = self.conv0(x, edge_index)
        # x = self.norm0(x, batch)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.drop, training=self.training)
        # print(x.dtype)
        # print(edge_index.dtype)
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        # print(x.dtype)
        # print(edge_index.dtype)
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # 2. Readout layer
        # x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.topk_pool(x, edge_index, batch=batch)[0]
        x = x.view(batch[-1] + 1, -1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin0(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        z = x  # extract last layer features
        x = self.lin(x)

        return x, z
    

######################################################################################
# CNN + GAT model
######################################################################################

class Conv1d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,), dilation=(1,), 
                 if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.SELU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x
    

# first 2 layer with 3, kernel
class CNNblock(nn.Module):
    def __init__(self):
        super(CNNblock, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(num_features=256),
                                   nn.ReLU(),
                                   ) 
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(num_features=64),
                                   nn.ReLU(),
                                   )

    def forward(self, x):
        x = self.conv0(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.conv1(x)
        x = F.dropout(x, 0.3, training=self.training)
        
        return x
    

# first 2 layer with 1, kernel
class CNNblock1(nn.Module):
    def __init__(self) -> None:
        super(CNNblock1, self).__init__()
        self.conv0 = nn.Sequential(Conv1d(1024, 256, kernel_size=(1,), stride=1),
                                   Conv1d(256, 64, kernel_size=(1,), stride=1),
                                   )
        
    def forward(self, x):
        x = self.conv0(x)
        
        return x
    

class multiscale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(multiscale, self).__init__()

        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)

        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True)
        )

    def forward(self, x):

        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = torch.cat([x0, x1, x2, x3], dim=1) # batch_size * 4outchannel * len
        return x4 + x
    

######################################################################################
# CNN + GAT model
######################################################################################


class Net1(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv_s2v = CNNblock()
        self.conv_lm = CNNblock()
        
        self.multiscale_s2v = multiscale(64, 16)
        self.multiscale_lm = multiscale(64, 16)
        # self.dpcnn = DPCNN(64 * 4, number_of_layers)
        
    def forward(self, s2v_embedding, lm_embedding):
        
        x0 = s2v_embedding # (N, 1024, 100)
        x1 = lm_embedding
        
        x0 = self.conv_s2v(x0)
        x1 = self.conv_lm(x1)
        
        x0 = self.multiscale_s2v(x0)
        x1 = self.multiscale_lm(x1)
        
        x = torch.cat([x0, x1], dim=1)

        return x
    
class Net2(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv_s2v = CNNblock()
        self.conv_lm1 = CNNblock()
        self.conv_lm2 = CNNblock()
        
        self.multiscale_s2v = multiscale(64, 16)
        self.multiscale_lm1 = multiscale(64, 16)
        self.multiscale_lm2 = multiscale(64, 16)

        
    def forward(self, s2v_embedding, lm_embedding):
        
        x0 = s2v_embedding # (N, 1024, 100)
        x1 = lm_embedding[0]
        x2 = lm_embedding[1]
        
        x0 = self.conv_s2v(x0)
        x1 = self.conv_lm1(x1)
        x2 = self.conv_lm2(x2)
        
        x0 = self.multiscale_s2v(x0)
        x1 = self.multiscale_lm1(x1)
        x2 = self.multiscale_lm1(x2)
        
        x = torch.cat([x0, x1, x2], dim=1)

        return x
    
# ! #################################################################################
    
class Net3(nn.Module):
    def __init__(self, lm_num):
        super().__init__()
        self.conv_s2v = CNNblock()
        self.multiscale_s2v = multiscale(64, 16)
        
        self.conv_lm = nn.ModuleList()
        self.multiscale_lm = nn.ModuleList()
        if lm_num != 0:
            for _ in range(lm_num):
                self.conv_lm.append(CNNblock())
                self.multiscale_lm.append(multiscale(64, 16))
        
    def forward(self, s2v_embedding, lm_embedding):
        
        x_list = []
        x0 = s2v_embedding # (N, 1024, 100)
        x0 = self.conv_s2v(x0)
        x0 = self.multiscale_s2v(x0)
        x_list.append(x0)
        
        if len(lm_embedding) != 0:
            if len(self.conv_lm) == 1:
                x = lm_embedding
                conv_layer = self.conv_lm[0]
                multiscale_layer = self.multiscale_lm[0]
                x = conv_layer(x)
                x = multiscale_layer(x)
                x_list.append(x)
                
            else:
                for i in range(len(self.conv_lm)):
                    x = lm_embedding[i]
                    conv_layer = self.conv_lm[i]
                    multiscale_layer = self.multiscale_lm[i]
                    
                    x = conv_layer(x)
                    x = multiscale_layer(x)
                    x_list.append(x)
        
        x = torch.cat(x_list, dim=1)

        return x
    
    
class GATModel2(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, use_lm, nheads=1, k=10):
        super(GATModel2, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        if use_lm != 'both':
            self.CNN = Net1()
        else:
            self.CNN = Net2()
            node_feature_dim += 64

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._initialize_weights()
        
    def _initialize_weights(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)
    

    def forward(self, x, edge_index, s2v, lm, node_num, batch):
        # todo: concat s2v, lm to node feature
        e = self.CNN(s2v, lm) # num_batch, 128, 100
        
        f_list = []
        for i, n in enumerate(node_num):
            # print(i, n)
            f = e[i][ : , :n]
            f_list.append(f)
        
        emb = torch.cat(f_list, dim=1).permute(1,0) # num_node_batch, 128 
        x = torch.cat((x, emb), dim = 1) # num_node_batch, x.shape[1] + 128 
        
        # run GAT
        # edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.clone().detach()
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        # print(x.dtype)
        # print(edge_index.dtype)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # 2. Readout layer
        # x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.topk_pool(x, edge_index, batch=batch)[0]
        x = x.view(batch[-1] + 1, -1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin0(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        z = x  # extract last layer features
        x = self.lin(x)

        return x, z
    
# ! #################################################################################

# ! #################################################################################

class GATModel3(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, use_lm, nheads=1, k=10):
        super(GATModel3, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k
        self.lm_num = len(use_lm)

        self.CNN = Net3(self.lm_num)
        node_feature_dim += 64 * self.lm_num

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._initialize_weights()
        
    def _initialize_weights(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)
    

    def forward(self, x, edge_index, s2v, lm, node_num, batch):
        
        
        e = self.CNN(s2v, lm) # num_batch, 128, 100
        z0 = e
        
        f_list = []
        e_list = []
        for i, n in enumerate(node_num):
            # print(i, n)
            f = e[i][ : , :n]
            f_list.append(f)
            
            e_list.append(e[i])
        z1 = e_list
        
        emb = torch.cat(f_list, dim=1).permute(1,0) # [num_node_batch, 64 * n]
        x = torch.cat((x, emb), dim = 1) # [num_node_batch, x.shape[1] + 64 * n]
        
        # run GAT
        # edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.clone().detach()
        
        x = self.conv1(x, edge_index) # [num_node_batch, x.shape[1] + nhead * h]
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch) # [num_node_batch, h] 64

        z2 = x
        
        # 2. Readout layer
        # x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.topk_pool(x, edge_index, batch=batch)[0] # [1280, 64]
        x = x.view(batch[-1] + 1, -1) # [128, 640]

        z3 = x
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin0(x) # [128, 64]
        x = F.relu(x)
        
        x = self.lin1(x)
        x = F.relu(x)
        z = x  # extract last layer features

        x = self.lin(x)

        return x, z, z0, z1, z2, z3
    
    
    
######################################################################################
# CNN + GAT model
######################################################################################

class GATModelMll(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, use_lm, nheads=1, k=10):
        super(GATModelMll, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        if use_lm != 'both':
            self.CNN = Net1()
        else:
            self.CNN = Net2()
            node_feature_dim += 64

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._initialize_weights()
        
    def _initialize_weights(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)
    

    def forward(self, x, edge_index, s2v, lm, node_num, batch):
        # todo: concat s2v, lm to node feature
        e = self.CNN(s2v, lm) # num_batch, 128, 100
        
        f_list = []
        for i, n in enumerate(node_num):
            # print(i, n)
            f = e[i][ : , :n]
            f_list.append(f)
        
        emb = torch.cat(f_list, dim=1).permute(1,0) # num_node_batch, 128 
        x = torch.cat((x, emb), dim = 1) # num_node_batch, x.shape[1] + 128 
        
        # run GAT
        # edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.clone().detach()
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        # print(x.dtype)
        # print(edge_index.dtype)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # 2. Readout layer
        # x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.topk_pool(x, edge_index, batch=batch)[0]
        x = x.view(batch[-1] + 1, -1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin0(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        z = x  # extract last layer features
        x = self.lin(x)
        x = torch.sigmoid(x)

        return x, z