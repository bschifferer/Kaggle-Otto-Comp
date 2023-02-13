import logging
from typing import Dict, Iterable, Optional

import torch

import torch

def to_sparse_tensor(values_offset, seq_limit, device):
    """
    Create a sparse representation of the input tensor.
    values_offset is either a tensor or a tuple of tensor, offset.
    """
    values, offsets, diff_offsets, num_rows = _pull_values_offsets(values_offset)
    max_seq_len = _get_max_seq_len(diff_offsets)
    if max_seq_len > seq_limit:
        raise ValueError(
            "The default sequence length has been configured "
            + f"to {seq_limit} but the "
            + f"largest sequence in this batch have {max_seq_len} length"
        )
    return _build_sparse_tensor(values, offsets, diff_offsets, num_rows, seq_limit, device)

def _get_max_seq_len(diff_offsets):
    return int(diff_offsets.max())

def _pull_values_offsets(values_offset):
    # pull_values_offsets, return values offsets diff_offsets
    if isinstance(values_offset, tuple):
        values = values_offset[0].flatten()
        offsets = values_offset[1].flatten()
    else:
        values = values_offset.flatten()
        offsets = torch.arange(values.size()[0])
    num_rows = len(offsets)
    offsets = torch.cat([offsets, torch.cuda.LongTensor([len(values)])])
    diff_offsets = offsets[1:] - offsets[:-1]
    return values, offsets, diff_offsets, num_rows

def _get_indices(offsets, diff_offsets, device):
    row_ids = torch.arange(len(offsets) - 1, device=device)
    row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
    col_ids = torch.arange(len(row_offset_repeated), device=device) - row_offset_repeated
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices

def _get_sparse_tensor(values, indices, num_rows, seq_limit):
    sparse_tensor = torch.sparse_coo_tensor(
        indices.T, values, torch.Size([num_rows, seq_limit])
    )
    sparse_tensor = sparse_tensor.to_dense()
    return sparse_tensor

def _build_sparse_tensor(values, offsets, diff_offsets, num_rows, seq_limit, device):
    indices = _get_indices(offsets, diff_offsets, device)
    return _get_sparse_tensor(values, indices, num_rows, seq_limit)   