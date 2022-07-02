#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
from type_aggregator import EntityTypeAggregator, RelationTypeAggregator, Match

pi = 3.14159265358979323846
eps = 1e-6

query_name_dict = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }

def Identity(x):
    return x

class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers, with_regular=True):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer
        self.with_regular = with_regular

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)
        if self.with_regular == True:
            x = self.projection_regularizer(x)

        return x

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y

def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y

class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, with_regular=True):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        self.with_regular = with_regular
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        if self.with_regular:
            axis_embeddings = convert_to_axis(axis)
            arg_embeddings = convert_to_arg(arg)
        else:
            axis_embeddings = axis
            arg_embeddings = arg
        return axis_embeddings, arg_embeddings

class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings

class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0
        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi
        arg_embedding = pi - arg_embedding
        return axis_embedding, arg_embedding

def order_bounds(embedding):  # ensure lower < upper truth bound for logic embedding
    embedding = torch.clamp(embedding, 0, 1)
    lower, upper = torch.chunk(embedding, 2, dim=-1)
    contra = lower > upper
    if contra.any():  # contradiction
        mean = (lower + upper) / 2
        lower = torch.where(lower > upper, mean, lower)
        upper = torch.where(lower > upper, mean, upper)
    ordered_embedding = torch.cat([lower, upper], dim=-1)
    return ordered_embedding

def valclamp(x, a=1, b=6, lo=0, hi=1):  # relu1 with gradient-transparent clamp on negative
    elu_neg = a * (torch.exp(b * x) - 1)
    return ((x < lo).float() * (lo + elu_neg - elu_neg.detach()) +
            (lo <= x).float() * (x <= hi).float() * x +
            (hi < x).float())

class LogicIntersection(nn.Module):

    def __init__(self, dim, tnorm, bounded, use_att, use_gtrans):
        super(LogicIntersection, self).__init__()
        self.dim = dim
        self.tnorm = tnorm
        self.bounded = bounded
        self.use_att = use_att
        self.use_gtrans = use_gtrans  # gradient transparency

        if use_att:  # use attention with weighted t-norm
            self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)

            if bounded:
                self.layer2 = nn.Linear(2 * self.dim, self.dim)  # same weight for bound pair
            else:
                self.layer2 = nn.Linear(2 * self.dim, 2 * self.dim)

            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        if self.use_att:  # use attention with weighted t-norm
            layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, batch_size, 2 * dim)
            attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, batch_size, dim)
            attention = attention / torch.max(attention, dim=0, keepdim=True).values

            if self.bounded:  # same weight for bound pair
                attention = torch.cat([attention, attention], dim=-1)

            if self.tnorm == 'mins':  # minimum / Godel t-norm
                smooth_param = -10  # smooth minimum
                min_weights = attention * torch.exp(smooth_param * embeddings)
                embedding = torch.sum(min_weights * embeddings, dim=0) / torch.sum(min_weights, dim=0)
                if self.bounded:
                    embedding = order_bounds(embedding)

            elif self.tnorm == 'luk':  # Lukasiewicz t-norm
                embedding = 1 - torch.sum(attention * (1 - embeddings), dim=0)
                if self.use_gtrans:
                    embedding = valclamp(embedding, b=6. / embedding.shape[0])
                else:
                    embedding = torch.clamp(embedding, 0, 1)

            elif self.tnorm == 'prod':  # product t-norm
                embedding = torch.prod(torch.pow(torch.clamp(embeddings, 0, 1) + eps, attention), dim=0)

        else:  # no attention
            if self.tnorm == 'mins':  # minimum / Godel t-norm
                smooth_param = -10  # smooth minimum
                min_weights = torch.exp(smooth_param * embeddings)
                embedding = torch.sum(min_weights * embeddings, dim=0) / torch.sum(min_weights, dim=0)
                if self.bounded:
                    embedding = order_bounds(embedding)

            elif self.tnorm == 'luk':  # Lukasiewicz t-norm
                embedding = 1 - torch.sum(1 - embeddings, dim=0)
                if self.use_gtrans:
                    embedding = valclamp(embedding, b=6. / embedding.shape[0])
                else:
                    embedding = torch.clamp(embedding, 0, 1)

            elif self.tnorm == 'prod':  # product t-norm
                embedding = torch.prod(embeddings, dim=0)

        return embedding

class LogicProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, num_layers, bounded, with_sigmoid=False):
        super(LogicProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bounded = bounded
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        self.with_sigmoid = with_sigmoid
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        if self.with_sigmoid:
            x = torch.sigmoid(x)

        if self.bounded:
            lower, upper = torch.chunk(x, 2, dim=-1)
            upper = lower + upper * (1 - lower)
            x = torch.cat([lower, upper], dim=-1)

        return x

class SizePredict(nn.Module):
    def __init__(self, entity_dim):
        super(SizePredict, self).__init__()

        self.layer2 = nn.Linear(entity_dim, entity_dim // 4)
        self.layer1 = nn.Linear(entity_dim // 4, entity_dim // 16)
        self.layer0 = nn.Linear(entity_dim // 16, 1)

        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer0.weight)

    def forward(self, entropy_embedding):
        x = self.layer2(entropy_embedding)
        x = F.relu(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer0(x)
        x = torch.sigmoid(x)

        return x.squeeze()

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, ntype, hidden_dim, entity2type, relation2type, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, center_reg=None, logic_mode=None, model_mode='baseline', drop=0., neighbor_ent_type_samples=32):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.ntype = ntype
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.entity2type = torch.tensor(entity2type).cuda()
        self.relation2type = torch.tensor(relation2type).cuda()
        self.neighbor_ent_type_samples = neighbor_ent_type_samples
        if self.geo == 'vec' or self.geo == 'box' or self.geo == 'cone':
            self.ent_neighbor_type_agg = EntityTypeAggregator(input_dim=hidden_dim, output_dim=hidden_dim, self_included=True, neighbor_ent_type_samples=self.neighbor_ent_type_samples)
        elif self.geo == 'beta' or self.geo == 'logic':
            self.ent_neighbor_type_agg = EntityTypeAggregator(input_dim=hidden_dim*2, output_dim=hidden_dim*2, self_included=True, neighbor_ent_type_samples=self.neighbor_ent_type_samples)

        if self.geo == 'logic':
            self.rel_neighbor_type_agg = RelationTypeAggregator(hidden_dim * 2)
        else:
            self.rel_neighbor_type_agg = RelationTypeAggregator(hidden_dim)
        if self.geo == 'logic':
            self.match = Match(hidden_dim * 2)
        else:
            self.match = Match(hidden_dim)
        self.model_mode = model_mode

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.type_dim = hidden_dim

        self.cen = center_reg
        
        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # centor for entities
            activation, cen = box_mode
            self.cen = cen # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # center for entities
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
        elif self.geo == 'cone':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=True)  # axis for entities
            self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]
            self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)
            self.axis_scale = 1.0
            self.arg_scale = 1.0
        elif self.geo == 'logic':
            self.tnorm, self.bounded, use_att, use_gtrans, hidden_dim, num_layers = logic_mode
            if self.bounded:
                lower = torch.rand((nentity, self.entity_dim))
                upper = lower + torch.rand((nentity, self.entity_dim)) * (1 - lower)
                self.entity_embedding = nn.Parameter(torch.cat([lower, upper], dim=-1))
            else:
                self.entity_embedding = nn.Parameter(torch.rand((nentity, self.entity_dim * 2)))

        if self.geo in ['box', 'vec', 'beta', 'cone']:
            nn.init.uniform_(
                tensor=self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if self.geo == 'beta' or self.geo == 'logic':
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim * 2))
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        if self.geo in ['box', 'vec', 'beta', 'cone']:
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding, 
                a=0., 
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)

            self.type_embedding = nn.Parameter(torch.zeros(self.ntype, self.type_dim))
            nn.init.uniform_(
                tensor=self.type_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
            self.type_embedding = nn.Parameter(torch.zeros(self.ntype, self.type_dim))
            nn.init.uniform_(
                tensor=self.type_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.relation_center_net = CenterIntersection(self.relation_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2,
                                                 self.relation_dim * 2,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)
            self.projection_without_net = BetaProjection(self.entity_dim * 2,
                                                         self.relation_dim * 2,
                                                         hidden_dim,
                                                         self.projection_regularizer,
                                                         num_layers, with_regular=False)
            self.type_embedding = nn.Parameter(torch.zeros(self.ntype, self.type_dim * 2))
            nn.init.uniform_(
                tensor=self.type_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif self.geo == 'cone':
            self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.axis_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.arg_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.cone_proj = ConeProjection(self.entity_dim, 1600, 2)
            self.cone_intersection = ConeIntersection(self.entity_dim, drop)
            self.cone_negation = ConeNegation()
            self.relation_center_net = CenterIntersection(self.entity_dim)
            self.cone_without_proj = ConeProjection(self.entity_dim, 1600, 2, with_regular=False)
            self.type_embedding = nn.Parameter(torch.zeros(self.ntype, self.type_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.type_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif self.geo == 'logic':
            tnorm, bounded, use_att, use_gtrans, hidden_dim, num_layers = logic_mode
            self.center_net = LogicIntersection(self.entity_dim, tnorm, bounded, use_att, use_gtrans)
            self.relation_center_net = CenterIntersection(self.entity_dim * 2)
            self.projection_net = LogicProjection(self.entity_dim * 2,
                                             self.relation_dim * 2,
                                             hidden_dim,
                                             num_layers,
                                             bounded, with_sigmoid=True)
            self.projection_without_net = LogicProjection(self.entity_dim,
                                                          self.relation_dim,
                                                          hidden_dim,
                                                          num_layers,
                                                          bounded, with_sigmoid=False)
            self.type_embedding = nn.Parameter(torch.rand(self.ntype, self.type_dim * 2))

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'cone':
            return self.forward_cone(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'logic':
            return self.forward_logic(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                if self.model_mode == 'baseline':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                elif self.model_mode == 'temp':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                    ent_type_id = torch.index_select(self.entity2type, dim=0, index=queries[:, idx])
                    entity_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=ent_type_id.view(-1)).view(ent_type_id.shape[0], ent_type_id.shape[1], -1)
                    embedding = self.ent_neighbor_type_agg(embedding, entity_neighbor_type_embedding)

                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    if self.model_mode == 'baseline':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        embedding += r_embedding
                    elif self.model_mode == 'temp':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        rel_type_id = torch.index_select(self.relation2type, dim=0, index=queries[:, idx])
                        relation_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=rel_type_id.view(-1)).view(rel_type_id.shape[0], rel_type_id.shape[1], -1)
                        relation_neighbor_type_embedding = torch.transpose(relation_neighbor_type_embedding, 0, 1)
                        rel_type_cent = self.center_net(relation_neighbor_type_embedding)

                        embedding = embedding.unsqueeze(1)
                        r_embedding = r_embedding.unsqueeze(1)
                        rel_type_cent = rel_type_cent.unsqueeze(1)
                        ent_rel = self.match([embedding, r_embedding])
                        ent_rel_type = self.match([embedding, rel_type_cent])
                        rel_ent = self.match([r_embedding, embedding])
                        rel_rel_type = self.match([r_embedding, rel_type_cent])

                        embedding = self.rel_neighbor_type_agg([ent_rel.squeeze(1), ent_rel_type.squeeze(1)])
                        r_embedding = self.rel_neighbor_type_agg([rel_ent.squeeze(1), rel_rel_type.squeeze(1)])
                        embedding += r_embedding

                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                if self.model_mode == 'baseline':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                elif self.model_mode == 'temp':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                    ent_type_id = torch.index_select(self.entity2type, dim=0, index=queries[:, idx])
                    entity_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=ent_type_id.view(-1)).view(ent_type_id.shape[0], ent_type_id.shape[1], -1)
                    embedding = self.ent_neighbor_type_agg(embedding, entity_neighbor_type_embedding)

                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    if self.model_mode == 'baseline':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                        embedding += r_embedding
                        offset_embedding += self.func(r_offset_embedding)
                    elif self.model_mode == 'temp':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                        rel_type_id = torch.index_select(self.relation2type, dim=0, index=queries[:, idx])
                        relation_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=rel_type_id.view(-1)).view(rel_type_id.shape[0], rel_type_id.shape[1], -1)
                        relation_neighbor_type_embedding = torch.transpose(relation_neighbor_type_embedding, 0, 1)
                        rel_type_cent = self.center_net(relation_neighbor_type_embedding)

                        embedding = embedding.unsqueeze(1)
                        r_embedding = r_embedding.unsqueeze(1)
                        rel_type_cent = rel_type_cent.unsqueeze(1)
                        ent_rel = self.match([embedding, r_embedding])
                        ent_rel_type = self.match([embedding, rel_type_cent])
                        rel_ent = self.match([r_embedding, embedding])
                        rel_rel_type = self.match([r_embedding, rel_type_cent])

                        embedding = self.rel_neighbor_type_agg([ent_rel.squeeze(1), ent_rel_type.squeeze(1)])
                        r_embedding = self.rel_neighbor_type_agg([rel_ent.squeeze(1), rel_rel_type.squeeze(1)])

                        embedding += r_embedding
                        offset_embedding += self.func(r_offset_embedding)

                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_beta(self, queries, query_structure, idx, filter_flag=False):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        if self.model_mode == 'temp':
            if filter_flag == False:
                if query_structure in query_name_dict and query_name_dict[query_structure] == 'ip':
                    filter_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                if self.model_mode == 'baseline':
                    embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                elif self.model_mode == 'temp':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                    ent_type_id = torch.index_select(self.entity2type, dim=0, index=queries[:, idx])
                    entity_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=ent_type_id.view(-1)).view(ent_type_id.shape[0], ent_type_id.shape[1], -1)
                    embedding = self.ent_neighbor_type_agg(embedding, entity_neighbor_type_embedding)

                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    if self.model_mode == 'temp':
                        embedding = self.entity_regularizer(embedding)
                    embedding = 1./embedding
                else:
                    if self.model_mode == 'baseline':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        embedding = self.projection_net(embedding, r_embedding)
                    elif self.model_mode == 'temp':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        rel_type_id = torch.index_select(self.relation2type, dim=0, index=queries[:, idx])
                        relation_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=rel_type_id.view(-1)).view(rel_type_id.shape[0], rel_type_id.shape[1], -1)
                        relation_neighbor_type_embedding = torch.transpose(relation_neighbor_type_embedding, 0, 1)
                        alpha_relation_neighbor_type_embedding, beta_relation_neighbor_type_embedding = torch.chunk(relation_neighbor_type_embedding, 2, dim=-1)
                        alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
                        alpha_r_embedding, beta_r_embedding = torch.chunk(r_embedding, 2, dim=-1)
                        alpha_rel_type_cent = self.relation_center_net(alpha_relation_neighbor_type_embedding)
                        beta_rel_type_cent = self.relation_center_net(beta_relation_neighbor_type_embedding)

                        alpha_embedding = alpha_embedding.unsqueeze(1)
                        alpha_r_embedding = alpha_r_embedding.unsqueeze(1)
                        alpha_rel_type_cent = alpha_rel_type_cent.unsqueeze(1)
                        alpha_ent_rel = self.match([alpha_embedding, alpha_r_embedding])
                        alpha_ent_rel_type = self.match([alpha_embedding, alpha_rel_type_cent])
                        alpha_rel_ent = self.match([alpha_r_embedding, alpha_embedding])
                        alpha_rel_rel_type = self.match([alpha_r_embedding, alpha_rel_type_cent])

                        alpha_embedding = self.rel_neighbor_type_agg([alpha_ent_rel.squeeze(1), alpha_ent_rel_type.squeeze(1)])
                        alpha_r_embedding = self.rel_neighbor_type_agg([alpha_rel_ent.squeeze(1), alpha_rel_rel_type.squeeze(1)])

                        beta_embedding = beta_embedding.unsqueeze(1)
                        beta_r_embedding = beta_r_embedding.unsqueeze(1)
                        beta_rel_type_cent = beta_rel_type_cent.unsqueeze(1)
                        beta_ent_rel = self.match([beta_embedding, beta_r_embedding])
                        beta_ent_rel_type = self.match([beta_embedding, beta_rel_type_cent])
                        beta_rel_ent = self.match([beta_r_embedding, beta_embedding])
                        beta_rel_rel_type = self.match([beta_r_embedding, beta_rel_type_cent])

                        beta_embedding = self.rel_neighbor_type_agg([beta_ent_rel.squeeze(1), beta_ent_rel_type.squeeze(1)])
                        beta_r_embedding = self.rel_neighbor_type_agg([beta_rel_ent.squeeze(1), beta_rel_rel_type.squeeze(1)])

                        embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
                        r_embedding = torch.cat([alpha_r_embedding, beta_r_embedding], dim=-1)
                        embedding = self.projection_without_net(embedding, r_embedding)
                idx += 1
            if self.model_mode == 'temp':
                if filter_flag == False:
                    embedding = self.entity_regularizer(embedding)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def embed_query_cone(self, queries, query_structure, idx, filter_flag=False):
        all_relation_flag = True
        if self.model_mode == 'temp':
            if filter_flag == False:
                if query_structure in query_name_dict and query_name_dict[query_structure] == 'ip':
                    filter_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                if self.model_mode == 'baseline':
                    axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                    axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                    axis_entity_embedding = convert_to_axis(axis_entity_embedding)
                elif self.model_mode == 'temp':
                    axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                    ent_type_id = torch.index_select(self.entity2type, dim=0, index=queries[:, idx])
                    entity_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=ent_type_id.view(-1)).view(ent_type_id.shape[0], ent_type_id.shape[1], -1)
                    axis_entity_embedding = self.ent_neighbor_type_agg(axis_entity_embedding, entity_neighbor_type_embedding)
                    axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[0], idx, filter_flag)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    if self.model_mode == 'temp':
                        axis_embedding = self.angle_scale(axis_embedding, self.axis_scale)
                        axis_embedding = convert_to_axis(axis_embedding)
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)

                # projection
                else:
                    if self.model_mode == 'baseline':
                        axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                        arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])

                        axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                        arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

                        axis_r_embedding = convert_to_axis(axis_r_embedding)
                        arg_r_embedding = convert_to_axis(arg_r_embedding)

                        axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding, arg_r_embedding)
                    elif self.model_mode == 'temp':
                        axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                        axis_rel_type_id = torch.index_select(self.relation2type, dim=0, index=queries[:, idx])
                        axis_relation_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=axis_rel_type_id.view(-1)).view(axis_rel_type_id.shape[0], axis_rel_type_id.shape[1], -1)
                        axis_relation_neighbor_type_embedding = torch.transpose(axis_relation_neighbor_type_embedding, 0, 1)
                        axis_rel_type_cent = self.relation_center_net(axis_relation_neighbor_type_embedding)
                        axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                        axis_rel_type_cent = self.angle_scale(axis_rel_type_cent, self.axis_scale)

                        arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                        arg_rel_type_id = torch.index_select(self.relation2type, dim=0, index=queries[:, idx])
                        arg_relation_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=arg_rel_type_id.view(-1)).view(arg_rel_type_id.shape[0], arg_rel_type_id.shape[1], -1)
                        arg_relation_neighbor_type_embedding = torch.transpose(arg_relation_neighbor_type_embedding, 0, 1)
                        arg_rel_type_cent = self.relation_center_net(arg_relation_neighbor_type_embedding)
                        arg_r_embedding = self.angle_scale(arg_r_embedding, self.axis_scale)
                        arg_rel_type_cent = self.angle_scale(arg_rel_type_cent, self.axis_scale)

                        axis_embedding = axis_embedding.unsqueeze(1)
                        axis_r_embedding = axis_r_embedding.unsqueeze(1)
                        axis_rel_type_cent = axis_rel_type_cent.unsqueeze(1)
                        axis_ent_rel = self.match([axis_embedding, axis_r_embedding])
                        axis_ent_rel_type = self.match([axis_embedding, axis_rel_type_cent])
                        axis_rel_ent = self.match([axis_r_embedding, axis_embedding])
                        axis_rel_rel_type = self.match([axis_r_embedding, axis_rel_type_cent])

                        axis_embedding = self.rel_neighbor_type_agg([axis_ent_rel.squeeze(1), axis_ent_rel_type.squeeze(1)])
                        axis_r_embedding = self.rel_neighbor_type_agg([axis_rel_ent.squeeze(1), axis_rel_rel_type.squeeze(1)])

                        arg_embedding = arg_embedding.unsqueeze(1)
                        arg_r_embedding = arg_r_embedding.unsqueeze(1)
                        arg_rel_type_cent = arg_rel_type_cent.unsqueeze(1)
                        arg_ent_rel = self.match([arg_embedding, arg_r_embedding])
                        arg_ent_rel_type = self.match([arg_embedding, arg_rel_type_cent])
                        arg_rel_ent = self.match([arg_r_embedding, arg_embedding])
                        arg_rel_rel_type = self.match([arg_r_embedding, arg_rel_type_cent])

                        arg_embedding = self.rel_neighbor_type_agg([arg_ent_rel.squeeze(1), arg_ent_rel_type.squeeze(1)])
                        arg_r_embedding = self.rel_neighbor_type_agg([arg_rel_ent.squeeze(1), arg_rel_rel_type.squeeze(1)])

                        axis_embedding, arg_embedding = self.cone_without_proj(axis_embedding, arg_embedding, axis_r_embedding, arg_r_embedding)

                idx += 1
            if self.model_mode == 'temp':
                if filter_flag == False:
                    axis_embedding = convert_to_axis(axis_embedding)
                    arg_embedding = convert_to_axis(arg_embedding)
        else:
            # intersection
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[i], idx, filter_flag)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def embed_query_logic(self, queries, query_structure, idx, filter_flag=False):
        '''
        Iterative embed a batch of queries with same structure using logic embeddings
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        if filter_flag == False:
            if query_structure in query_name_dict and query_name_dict[query_structure] == 'ip':
                filter_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                if self.model_mode == 'baseline':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                elif self.model_mode == 'temp':
                    embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                    ent_type_id = torch.index_select(self.entity2type, dim=0, index=queries[:, idx])
                    entity_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=ent_type_id.view(-1)).view(ent_type_id.shape[0], ent_type_id.shape[1], -1)
                    embedding = self.ent_neighbor_type_agg(embedding, entity_neighbor_type_embedding)

                idx += 1
            else:
                embedding, idx = self.embed_query_logic(queries, query_structure[0], idx, filter_flag)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    if self.bounded:
                        lower_embedding, upper_embedding = torch.chunk(embedding, 2, dim=-1)
                        embedding = torch.cat([1 - upper_embedding, 1 - lower_embedding], dim=-1)
                    else:
                        embedding = 1 - embedding
                else:
                    if self.model_mode == 'baseline':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        embedding = self.projection_net(embedding, r_embedding)
                    elif self.model_mode == 'temp':
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        rel_type_id = torch.index_select(self.relation2type, dim=0, index=queries[:, idx])
                        relation_neighbor_type_embedding = torch.index_select(self.type_embedding, dim=0, index=rel_type_id.view(-1)).view(rel_type_id.shape[0], rel_type_id.shape[1], -1)
                        relation_neighbor_type_embedding = torch.transpose(relation_neighbor_type_embedding, 0, 1)
                        rel_type_cent = self.relation_center_net(relation_neighbor_type_embedding)

                        embedding = embedding.unsqueeze(1)
                        r_embedding = r_embedding.unsqueeze(1)
                        rel_type_cent = rel_type_cent.unsqueeze(1)
                        ent_rel = self.match([embedding, r_embedding])
                        ent_rel_type = self.match([embedding, rel_type_cent])
                        rel_ent = self.match([r_embedding, embedding])
                        rel_rel_type = self.match([r_embedding, rel_type_cent])

                        embedding = self.rel_neighbor_type_agg([ent_rel.squeeze(1), ent_rel_type.squeeze(1)])
                        r_embedding = self.rel_neighbor_type_agg([rel_ent.squeeze(1), rel_rel_type.squeeze(1)])
                        embedding = self.projection_net(embedding, r_embedding)

                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_logic(queries, query_structure[i], idx, filter_flag)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))
    
    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                                                self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, None

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, None

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(query_structure),
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure],
                                                                           query_structure,
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0] // 2, 2, 1,
                                                                         -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0] // 2, 2, 1,
                                                                       -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[
                    all_idxs]  # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[
                    all_union_idxs]  # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(
                        1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(
                        batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(
                        batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, None

    def cal_logit_cone(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        logit = self.gamma - distance * self.modulus

        return logit

    def forward_cone(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_axis_embeddings, all_arg_embeddings = [], [], []
        all_union_idxs, all_union_axis_embeddings, all_union_arg_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding, _ = \
                    self.embed_query_cone(self.transform_union_query(batch_queries_dict[query_structure], query_structure), self.transform_union_structure(query_structure), 0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
            else:
                axis_embedding, arg_embedding, _ = self.embed_query_cone(batch_queries_dict[query_structure], query_structure, 0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)

        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)


            if len(all_union_axis_embeddings) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_union_logit = self.cal_logit_cone(positive_embedding, all_union_axis_embeddings, all_union_arg_embeddings)

                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, None

    def cal_logit_logic(self, entity_embedding, query_embedding):
        if self.bounded:
            lower_embedding, upper_embedding = torch.chunk(entity_embedding, 2, dim=-1)
            query_lower_embedding, query_upper_embedding = torch.chunk(query_embedding, 2, dim=-1)

            lower_dist = torch.norm(lower_embedding - query_lower_embedding, p=1, dim=-1)
            upper_dist = torch.norm(query_upper_embedding - upper_embedding, p=1, dim=-1)

            logit = self.gamma - (lower_dist + upper_dist) / 2 / lower_embedding.shape[-1]
        else:
            logit = self.gamma - torch.norm(entity_embedding - query_embedding, p=1, dim=-1) / query_embedding.shape[-1]

        logit *= 100

        return logit

    def forward_logic(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_entropy = None
        all_idxs, all_embeddings = [], []
        all_union_idxs, all_union_embeddings = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                embedding, _ = \
                    self.embed_query_logic(self.transform_union_query(batch_queries_dict[query_structure],
                                                                      query_structure),
                                           self.transform_union_structure(query_structure),
                                           0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_embeddings.append(embedding)
            else:
                embedding, _ = self.embed_query_logic(batch_queries_dict[query_structure],
                                                      query_structure,
                                                      0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_embeddings.append(embedding)

        if len(all_embeddings) > 0:
            all_embeddings = torch.cat(all_embeddings, dim=0).unsqueeze(1)

            if positive_sample is None and self.bounded:  # test step - measure entropy
                lower, upper = torch.chunk(all_embeddings, 2, dim=-1)
                truth_interval = upper - lower
                distribution = torch.distributions.uniform.Uniform(lower, upper + eps)
                all_entropy = (distribution.entropy(), truth_interval)

        if len(all_union_embeddings) > 0:
            all_union_embeddings = torch.cat(all_union_embeddings, dim=0).unsqueeze(1)
            all_union_embeddings = all_union_embeddings.view(all_union_embeddings.shape[0] // 2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]  # positive samples for non-union queries in this batch
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_logic(positive_embedding, all_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]  # positive samples for union queries in this batch
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_logic(positive_embedding, all_union_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_logic(negative_embedding, all_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_logic(negative_embedding, all_union_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, all_entropy

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs, _ = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics