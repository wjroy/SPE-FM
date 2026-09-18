import math

import torch
from torch import nn

from .vendor.informer.model import Informer
from .vendor.patchtst.PatchTST_backbone import PatchTST_backbone


class AdditivePool(nn.Module):
    def __init__(self, width, attention_width):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, attention_width), nn.Tanh(),
                                   nn.Linear(attention_width, 1, bias=False))

    def forward(self, x):
        weights = self.score(x).squeeze(-1).softmax(dim=-1).unsqueeze(-1)
        return (weights*x).sum(dim=1)


def sinusoidal(length, width):
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[:width // 2])
    return encoding[None]


class AttBiLSTM(nn.Module):
    def __init__(self, horizon, hidden_size, layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden_size, layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.)
        self.pool = AdditivePool(2 * hidden_size, hidden_size)
        self.head = nn.Linear(2 * hidden_size, horizon)

    def forward(self, x):
        batch, length, nodes = x.shape
        sequence = x.transpose(1, 2).reshape(batch * nodes, length, 1)
        sequence, _ = self.rnn(sequence)
        return self.head(self.pool(sequence)).reshape(batch, nodes, -1).transpose(1, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, context, horizon, d_model, layers, nhead, dropout,
                 d_ff, variant='Transformer'):
        super().__init__()
        if variant != 'Transformer':
            raise ValueError('Unsupported temporal encoder variant.')
        self.context, self.variant = context, variant
        self.embedding = nn.Linear(1, d_model)
        self.register_buffer('position', sinusoidal(context, d_model))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout,
                                            activation='gelu', batch_first=True,
                                            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.Identity()
        self.pool = AdditivePool(d_model, d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        batch, length, nodes = x.shape
        if length != self.context:
            raise ValueError('Input length differs from the configured context.')
        sequence = x.transpose(1, 2).reshape(batch * nodes, length)
        tokens = sequence.unsqueeze(-1)
        hidden = self.dropout(self.embedding(tokens) + self.position)
        result = self.head(self.pool(self.norm(self.encoder(hidden))))
        return result.reshape(batch, nodes, -1).transpose(1, 2)


class PatchTSTBaseline(nn.Module):
    def __init__(self, context, horizon, nodes, parameters):
        super().__init__()
        p = parameters
        if min(context, horizon, nodes, p['layers'], p['d_model'], p['nhead']) < 1:
            raise ValueError('PatchTST dimensions must be positive.')
        if not 1 <= p['patch_length'] <= context or p['stride'] < 1 or p['d_model'] % p['nhead']:
            raise ValueError('Invalid PatchTST patch dimensions or attention heads.')
        if (p['head_type'], p['individual'], p['norm'], p['padding_patch']) != ('flatten', False, 'BatchNorm', 'end'):
            raise ValueError('Unsupported PatchTST head, normalization or padding.')
        self.context, self.horizon, self.nodes = context, horizon, nodes
        self.model = PatchTST_backbone(
            c_in=nodes, context_window=context, target_window=horizon,
            patch_len=p['patch_length'], stride=p['stride'], n_layers=p['layers'],
            d_model=p['d_model'], n_heads=p['nhead'], d_ff=p['d_ff'], norm=p['norm'],
            attn_dropout=p['attention_dropout'], dropout=p['dropout'], act=p['activation'],
            res_attention=p['residual_attention'], pre_norm=p['pre_norm'],
            pe=p['position_encoding'], learn_pe=p['learn_position'],
            fc_dropout=p['fc_dropout'], head_dropout=p['head_dropout'],
            padding_patch=p['padding_patch'], head_type=p['head_type'], individual=p['individual'],
            revin=p['revin'], affine=p['revin_affine'], subtract_last=p['subtract_last'])

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.context, self.nodes) or x.shape[0] < 1:
            raise ValueError('PatchTST expects [batch, context, monitoring nodes].')
        return self.model(x.transpose(1, 2)).transpose(1, 2)


class InformerBaseline(nn.Module):
    def __init__(self, context, horizon, parameters):
        super().__init__()
        p = parameters
        self.context, self.horizon = context, horizon
        self.label_len = p['label_len']
        if context < 2 or horizon < 1 or not 1 <= self.label_len <= context:
            raise ValueError('Invalid Informer context, decoder prefix or horizon.')
        if max(context, self.label_len + horizon) > 5000:
            raise ValueError('Informer input exceeds the positional embedding limit.')
        if p['d_model'] % p['nhead'] or min(p['layers'], p['decoder_layers'], p['factor']) < 1:
            raise ValueError('Invalid Informer dimensions, layers or sampling factor.')
        if (p['attention'], p['embedding'], p['frequency'], p['time_features'],
                p['decoder_padding']) != ('ProbSparse', 'timeF', 'd', 'zeros', 0):
            raise ValueError('Informer uses settlement-only inputs and zero decoder padding.')
        if p['activation'] not in ('gelu', 'relu'):
            raise ValueError('Unsupported Informer activation.')
        encoded_length = context
        if p['distil']:
            for _ in range(p['layers'] - 1):
                encoded_length = (encoded_length + 1) // 2
        if encoded_length < 2:
            raise ValueError('The distilled encoder must retain at least two tokens.')
        self.model = Informer(
            enc_in=1, dec_in=1, c_out=1, seq_len=context, label_len=self.label_len,
            out_len=horizon, factor=p['factor'], d_model=p['d_model'], n_heads=p['nhead'],
            e_layers=p['layers'], d_layers=p['decoder_layers'], d_ff=p['d_ff'],
            dropout=p['dropout'], attn='prob', embed=p['embedding'], freq=p['frequency'],
            activation=p['activation'], output_attention=False, distil=p['distil'], mix=p['mix'])

    def inputs(self, x):
        if x.ndim != 3 or x.shape[1] != self.context or min(x.shape[0], x.shape[2]) < 1:
            raise ValueError('Informer expects [batch, context, nodes] settlement inputs.')
        batch, length, nodes = x.shape
        encoder = x.transpose(1, 2).reshape(batch * nodes, length, 1)
        decoder = torch.cat((encoder[:, -self.label_len:],
                             encoder.new_zeros(batch * nodes, self.horizon, 1)), dim=1)
        encoder_marks = encoder.new_zeros(batch * nodes, length, 3)
        decoder_marks = encoder.new_zeros(batch * nodes, self.label_len + self.horizon, 3)
        return encoder, encoder_marks, decoder, decoder_marks

    def forward(self, x):
        inputs = self.inputs(x)
        prediction = self.model(*inputs)
        return prediction.reshape(x.shape[0], x.shape[2], self.horizon).transpose(1, 2)
