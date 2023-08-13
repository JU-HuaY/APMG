import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, LeakyReLU
import numpy as np
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from math import pi as PI
from .linear import CustomizedLinear, gen_mask
EPS = 1e-6
# from data_pre.profile import lineprofile


class ne_sca_attention(Module):
    def __init__(self, hid_sca, out_sca):
        super().__init__()
        self.linear = Linear(hid_sca, out_sca)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node, edge):
        edge1 = self.linear(edge)
        scale = edge1.size(-1) ** -0.5
        att_mask = self.softmax(node * edge1 * scale)
        norm_mask = att_mask * (1 / torch.mean(att_mask))
        # print((1 / torch.mean(att_mask)))
        atten = norm_mask * edge1  # * edge1.shape[-1]
        atten_node = node * atten
        return atten_node

class cro_attention(Module):
    def __init__(self, hid_sca, hid_vec, out_sca, out_vec):
        super().__init__()
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, node_scalar, node_vector, edge_scalar, edge_vector):

        edge_scalar_atten = self.e2n_linear(edge_scalar).unsqueeze(-1)
        # scale = edge_scalar_atten.size(-1) ** -0.5
        node_att_mask = self.softmax(node_vector * edge_scalar_atten) * edge_scalar_atten.size(1)

        edge_vector_atten = self.n2e_linear(node_scalar).unsqueeze(-1)
        # scale2 = edge_vector_atten.size(-1) ** -0.5
        edge_vector = self.edge_vnlinear(edge_vector)
        # print((edge_vector * edge_vector_atten * scale).shape)
        edge_att_mask = self.softmax2(edge_vector * edge_vector_atten) * edge_vector_atten.size(1)

        # att_mask = (node_att_mask + edge_att_mask) * (edge_scalar_atten.size(1)/2)

        y_node_vector = node_att_mask * node_vector# * edge_scalar_atten
        y_edge_vector = edge_att_mask * edge_vector# * edge_vector_atten

        y_vector = y_node_vector + y_edge_vector

        return y_vector

class AttentionMPNNs(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = GVLinear(node_sca, node_vec, out_sca, out_vec)
        self.edge_gvp = GVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec)

        self.ne_attention = ne_sca_attention(hid_sca, out_sca)
        # self.cro_attention = cro_attention(hid_sca, hid_vec, out_sca, out_vec)
        # self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        # self.gru1 = nn.GRU(out_sca, out_sca, 1)

        self.out_gvlienar = GVLinear(out_sca, out_vec, out_sca, out_vec)

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=None, annealing=False):
        node_scalar, node_vector = self.node_gvlinear(node_features)

        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node] #new_point_letter = lenth_edge
        edge_scalar, edge_vector = self.edge_gvp(edge_features)
        # print(edge_features)
        # print(edge_index_node)
        y_scalar = self.ne_attention(node_scalar, edge_scalar)
        # y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector
        # y_vector = self.cro_attention(node_scalar, node_vector, edge_scalar, edge_vector)

        output = self.out_gvlienar((y_scalar, y_vector))
        # y1, y2 = output
        # print(y1.shape)
        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]   # (A, 1)

        return output


class GVPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = GVLinear(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        sca, vec = self.gv_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class GVLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vector = VNLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        # self.conv = nn.Conv1d(dim_hid, dim_hid, 3)
        # self.group_lin_vector = VNGroupLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_scalar = Conv1d(in_scalar + dim_hid, out_scalar, 1, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        # self.linear = Linear(3, 1, bias=False)
        # self.scalar_to_vector_gates = CustomizedLinear(out_scalar, out_vector, mask=gen_mask(out_scalar, out_vector, 0.2))
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)
        # self.lin_scalar = CustomizedLinear(in_scalar + dim_hid, out_scalar, bias=False, mask=gen_mask(in_scalar + dim_hid, out_scalar, 0.2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        # feat_vector_norm = self.conv(feat_vector_inter).squeeze(2)
        # print(feat_vector_inter)
        # feat_vector_norm = self.linear(feat_vector_inter).squeeze(dim=-1)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)  # (N_samples, dim_hid)
        feat_scalar_cat = torch.cat([feat_vector_norm, feat_scalar], dim=-1)  # (N_samples, dim_hid+in_scalar)

        out_scalar = self.lin_scalar(feat_scalar_cat)
        out_vector = self.lin_vector2(feat_vector_inter)

        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim=-1)
        out_vector = gating * out_vector
        # print(out_vector.shape)
        return out_scalar, out_vector


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x_out = self.map_to_feat(x.transpose(-2, -1)).transpose(-2, -1)
        return x_out

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        d = self.map_to_dir(x.transpose(-2,-1)).transpose(-2,-1)  # (N_samples, N_feat, 3)
        dotprod = (x*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        x_out = (self.negative_slope * x +
                (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d)))
        return x_out



def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (fr(input.real) + fi(input.imag)).type(dtype)


class ComplexLinear(nn.Module):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(ComplexLinear, self).__init__()
        self.FC_R = nn.Linear(in_channels, out_channels, *args, **kwargs)
        self.FC_I = VNLinear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x_out = apply_complex(self.FC_R, self.FC_I, x)
        # x_out = self.map_to_feat(x.transpose(-2,-1)).transpose(-2,-1)
        return x_out


class ComplexGVLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.linear = ComplexLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_vector = VNGroupLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_scalar = Conv1d(in_scalar + dim_hid, out_scalar, 1, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        features_real, features_imag = features
        features = torch.complex(features_real, features_imag)
        features = self.linear(features)
        features_real, features_imag = features.real, features.imag
        # feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        feat_imag_norm = torch.norm(features_imag, p=2, dim=-1)  # (N_samples, dim_hid)
        feat_real_cat = torch.cat([feat_imag_norm, features_real], dim=-1)  # (N_samples, dim_hid+in_scalar)

        y_out = torch.complex(feat_real_cat, features_imag)
        out = apply_complex(self.lin_scalar, self.lin_vector2, y_out)
        out_real, out_imag = out.real, out.imag

        # gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim=-1)
        scale = out_real.size(-1) ** -0.5
        gating = self.softmax(self.scalar_to_vector_gates(out_real) * scale).unsqueeze(dim=-1)
        out_imag = gating * out_imag

        return out_real, out_imag



class ComplexGVPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = ComplexGVLinear(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        gv_real, gv_imag = self.gv_linear(x)
        real = self.act_vec(gv_real)
        imag = self.act_sca(gv_imag)
        # out = torch.complex(real, imag)
        return real, imag

class ComplexMessageModule(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = ComplexGVLinear(node_sca, node_vec, out_sca, out_vec)
        self.edge_gvp = ComplexGVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec)

        self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gvlienar = ComplexGVLinear(out_sca, out_vec, out_sca, out_vec)

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=None, annealing=False):
        node = self.node_gvlinear(node_features)
        node_real, node_imag = node.real[edge_index_node], node.imag[edge_index_node]
        edge_real, edge_imag = self.edge_gvp(edge_features)

        y_real = node_real * self.sca_linear(edge_real)
        y_node_imag = self.e2n_linear(edge_real).unsqueeze(-1) * node_imag
        y_edge_imag = self.n2e_linear(node_real).unsqueeze(-1) * self.edge_vnlinear(edge_imag)
        y_imag = y_node_imag + y_edge_imag

        # y_out = torch.complex(y_real, y_imag)
        output = self.out_gvlienar((y_real, y_imag))

        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]   # (A, 1)
        return output
