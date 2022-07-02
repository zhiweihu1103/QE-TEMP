# -*- coding: utf-8 -*-
"""
@Time ： 2021/10/12 12:41
@Auth ： zhiweihu
"""
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional

class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, act, self_included, neighbor_ent_type_samples):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included
        self.neighbor_ent_type_samples = neighbor_ent_type_samples

    def forward(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        pass

class EntityTypeAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, act=lambda x: x, self_included=True, with_sigmoid=False, neighbor_ent_type_samples=32):
        super(EntityTypeAggregator, self).__init__(input_dim, output_dim, act, self_included, neighbor_ent_type_samples)
        self.proj_layer = HighwayNetwork(neighbor_ent_type_samples, 1, 2, activation=nn.Sigmoid())

        multiplier = 2 if self_included else 1
        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        self.with_sigmoid = with_sigmoid

    def _call(self, self_vectors, neighbor_vectors):
        neighbor_vectors = torch.transpose(neighbor_vectors, 1, 2)
        neighbor_vectors = self.proj_layer(neighbor_vectors)
        neighbor_vectors = torch.transpose(neighbor_vectors, 1, 2)
        neighbor_vectors = neighbor_vectors.squeeze(1)

        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])
            output = torch.cat([self_vectors, neighbor_vectors], dim=-1)
        output = self.layer(output)
        output = output.view([-1, self.output_dim])
        if self.with_sigmoid:
            output = torch.sigmoid(output)

        return self.act(output)

class HighwayNetwork(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               n_layers: int,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](inputs))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)

class Match(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False):
        super(Match, self).__init__()
        self.map_linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.trans_linear = nn.Linear(hidden_size, hidden_size)
        self.with_sigmoid = with_sigmoid

    def forward(self, inputs):
        proj_p, proj_q = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = torch.nn.functional.softmax(att_weights, dim=-1)
        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul], 2)
        output = self.map_linear(all_con)
        if self.with_sigmoid:
            output = torch.sigmoid(output)
        return output

class RelationTypeAggregator(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False):
        super(RelationTypeAggregator, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.with_sigmoid = with_sigmoid

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear2(q)
        lp = self.linear2(p)
        mid = nn.Sigmoid()(lq+lp)
        output = p * mid + q * (1-mid)
        output = self.linear(output)
        if self.with_sigmoid:
            output = torch.sigmoid(output)
        return output
