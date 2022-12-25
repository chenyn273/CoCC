# encoding=utf-8
from argparse import Namespace
from typing import List, Tuple

import torch
import logging
from common import FLOAT_TYPE
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logging.basicConfig(level=logging.INFO)


class BaseModel(nn.Module, ABC):
    def __init__(self, *args):
        super(BaseModel, self).__init__()

    @staticmethod
    @abstractmethod
    def prepare_model_params(args: Namespace) -> Namespace:
        """
        construct parameters for __init__ function from model args
        """
        pass

    @staticmethod
    def log_args(**kwargs):
        # remove unwanted keys
        unwanted_keys = ['self', '__class__']
        for key in unwanted_keys:
            if key in kwargs:
                kwargs.pop(key)

        logging.info("Create model using parameters:")
        for key, value in kwargs.items():
            logging.info("{}={}".format(key, value))

    @abstractmethod
    def init_pretrain_embeddings(self, freeze: bool):
        pass

    @classmethod
    def load(cls, model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        if isinstance(args, Namespace):
            model = cls(**vars(args))
        else:
            model = cls(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str, args: dict):
        logging.info('save model parameters to [%s]' % path)
        logging.info('model args:\n{}'.format(args))
        params = {
            'args': args,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool, batch_first: bool,
                 dropout: float):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_lens: List[int], enforce_sorted: bool = True) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        packed_input = pack_padded_sequence(x, x_lens, batch_first=self.rnn.batch_first, enforce_sorted=enforce_sorted)
        self.rnn.flatten_parameters()
        encodings, (last_state, last_cell) = self.rnn(packed_input)
        encodings, _ = pad_packed_sequence(encodings, batch_first=self.rnn.batch_first)
        return encodings, (last_state, last_cell)


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, bias: bool = True):
        super(LSTMCell, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size=input_size,
                                    hidden_size=hidden_size,
                                    bias=bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, h_tm1: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        h_t, cell_t = self.rnn_cell(x, h_tm1)
        return h_t, cell_t


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def init_bias(self, value: float):
        self.linear.bias.data.fill_(value)

    def forward(self, x: Tensor):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


def permute_lstm_output(encodings: torch.Tensor, last_state: torch.Tensor, last_cell: torch.Tensor):
    # (batch_size, sent_len, hidden_size)
    encodings = encodings.permute(1, 0, 2)
    # (batch_size, hidden_size * directions * #layers)
    last_state = torch.cat([s.squeeze(0) for s in last_state.split(1, dim=0)], dim=-1)
    last_cell = torch.cat([c.squeeze(0) for c in last_cell.split(1, dim=0)], dim=-1)
    return encodings, last_state, last_cell


def get_sent_masks(max_len: int, sent_lens: List[int], device: torch.device):
    src_sent_masks = torch.zeros(len(sent_lens), max_len, dtype=FLOAT_TYPE)
    for e_id, l in enumerate(sent_lens):
        # make all paddings to 1
        src_sent_masks[e_id, l:] = 1
    return src_sent_masks.to(device)
