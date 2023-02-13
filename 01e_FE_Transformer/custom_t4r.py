import logging
from typing import Dict, Iterable, Optional

import torch
import torchmetrics as tm


from transformers4rec.torch.block.base import Block, BuildableBlock, SequentialBlock
from transformers4rec.torch.block.mlp import MLPBlock
from transformers4rec.torch.masking import MaskedLanguageModeling
from transformers4rec.torch.ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt
from transformers4rec.torch.utils.torch_utils import LambdaModule
from transformers4rec.torch.block.base import BlockType
from transformers4rec.torch import PredictionTask
from transformers4rec.torch import Trainer

class CustomNextItemPredictionTask(PredictionTask):
    """This block performs item prediction task for session and sequential-based models.
    It requires a body containing a masking schema to use for training and target generation.
    For the supported masking schemes, please refers to:
    https://nvidia-merlin.github.io/Transformers4Rec/main/model_definition.html#sequence-masking
    Parameters
    ----------
    loss: torch.nn.Module
        Loss function to use. Defaults to NLLLos.
    metrics: Iterable[torchmetrics.Metric]
        List of ranking metrics to use for evaluation.
    task_block:
        Module to transform input tensor before computing predictions.
    task_name: str, optional
        Name of the prediction task, if not provided a name will be automatically constructed based
        on the target-name & class-name.
    weight_tying: bool
        The item id embedding table weights are shared with the prediction network layer.
    softmax_temperature: float
        Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
        Value 1.0 reduces to regular softmax.
    padding_idx: int
        pad token id.
    target_dim: int
        vocabulary size of item ids
    """

    DEFAULT_METRICS = (
        # default metrics suppose labels are int encoded
        NDCGAt(top_ks=[10, 20], labels_onehot=True),
        AvgPrecisionAt(top_ks=[10, 20], labels_onehot=True),
        RecallAt(top_ks=[10, 20], labels_onehot=True),
    )

    def __init__(
        self,
        loss = torch.nn.CrossEntropyLoss(),
        metrics: Iterable[tm.Metric] = DEFAULT_METRICS,
        task_block: Optional[BlockType] = None,
        task_name: str = "next-item",
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        padding_idx: int = 0,
        target_dim: int = None,
        item_probs = None,
        item_correction=False,
        neg_factor=0,
        temperature=1.0,
        remove_false_neg=False,
        item_correction_factor=1.0,
        loss_types=False,
        multi_task = 1,
        mt_tower = None,
        eval_task = 1,
        d_model = 1,
        use_tanh = False
    ):
        super().__init__(loss=loss, metrics=metrics, task_block=task_block, task_name=task_name)
        self.softmax_temperature = softmax_temperature
        self.weight_tying = weight_tying
        self.padding_idx = padding_idx
        self.target_dim = target_dim

        self.item_embedding_table = None
        self.masking = None
        self.item_probs = item_probs
        self.item_correction = item_correction
        self.neg_factor = neg_factor
        self.temperature = temperature
        self.remove_false_neg = remove_false_neg
        self.item_correction_factor=item_correction_factor
        self.loss_types = loss_types
        if use_tanh:
            self.type_emb = torch.nn.Embedding(4, d_model)
        else:
            self.type_emb = torch.nn.Embedding(4, multi_task)
        self.multi_task = multi_task
        self.mt_tower = mt_tower
        self.eval_task = torch.Tensor(eval_task).long()
        self.use_tanh = use_tanh

    def build(self, body, input_size, device=None, inputs=None, task_block=None, pre=None):
        """Build method, this is called by the `Head`."""
        if not len(input_size) == 3 or isinstance(input_size, dict):
            raise ValueError(
                "NextItemPredictionTask needs a 3-dim vector as input, found:" f"{input_size}"
            )

        # Retrieve the embedding module to get the name of itemid col and its related table
        if not inputs:
            inputs = body.inputs
        if not getattr(inputs, "item_id", None):
            raise ValueError(
                "For Item Prediction task a categorical_module "
                "including an item_id column is required."
            )
        self.embeddings = inputs.categorical_module
        if not self.target_dim:
            self.target_dim = self.embeddings.item_embedding_table.num_embeddings
        if self.weight_tying:
            self.item_embedding_table = self.embeddings.item_embedding_table
            item_dim = self.item_embedding_table.weight.shape[1]
            if input_size[-1] != item_dim and not task_block:
                LOG.warning(
                    f"Projecting inputs of NextItemPredictionTask to'{item_dim}' "
                    f"As weight tying requires the input dimension '{input_size[-1]}' "
                    f"to be equal to the item-id embedding dimension '{item_dim}'"
                )
                # project input tensors to same dimension as item-id embeddings
                task_block = MLPBlock([item_dim])

        # Retrieve the masking from the input block
        self.masking = inputs.masking
        if not self.masking:
            raise ValueError(
                "The input block should contain a masking schema for training and evaluation"
            )
        self.padding_idx = self.masking.padding_idx
        pre = CustomNextItemPredictionPrepareBlock(
            target_dim=self.target_dim,
            weight_tying=self.weight_tying,
            item_embedding_table=self.item_embedding_table,
            softmax_temperature=self.softmax_temperature,
        )
        super().build(
            body, input_size, device=device, inputs=inputs, task_block=task_block, pre=pre
        )

    def forward(self, inputs: torch.Tensor, targets=None, training=False, testing=False, **kwargs):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        x = inputs.float()

        if self.task_block:
            x = self.task_block(x)  # type: ignore

        # Retrieve labels from masking
        if training or testing:
            labels = self.masking.masked_targets  # type: ignore
            trg_flat = labels.flatten()
            
            non_pad_mask = trg_flat != self.padding_idx
            if self.remove_false_neg:
                num_idx = (labels != self.padding_idx).sum(axis=1)
                idx = torch.arange(labels.shape[0]).cuda()
                idx = torch.repeat_interleave(idx, num_idx)
                item_seq_idx = self.embeddings.item_seq[idx]
            else:
                item_seq_idx = None
            if self.loss_types:
                num_idx = (labels != self.padding_idx)
                types = self.embeddings.type_seq[num_idx]

            labels_all = torch.masked_select(trg_flat, non_pad_mask)

            # remove padded items, keep only masked positions
            x = self.remove_pad_3d(x, non_pad_mask)
            if self.multi_task>1:
                x_emb = self.type_emb(types)
                if self.use_tanh:
                    x_emb2 = torch.nn.Tanh()(x_emb)
                    x = x*x_emb2
                else:
                    x = torch.concat([x, x_emb], axis=1)
                    x = self.mt_tower(x)
            
            x, y = self.pre([x, 
                             labels_all, 
                             item_seq_idx,
                             self.item_probs,
                             self.item_correction,
                             self.neg_factor,
                             self.temperature,
                             self.item_correction_factor
                             ])  # type: ignore
            if self.loss_types:
                loss = self.loss(x, y, types)
            else:
                loss = self.loss(x, y)
            return {
                "loss": loss,
                "labels": labels_all,
                "predictions": x,
                # "pred_metadata": {},
                # "model_outputs": [],
            }
        else:
            # Get the hidden position to use for predicting the next item
            labels = self.embeddings.item_seq
            non_pad_mask = labels != self.padding_idx
            rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
            if isinstance(self.masking, MaskedLanguageModeling):
                last_item_sessions = non_pad_mask.sum(dim=1)
            else:
                last_item_sessions = non_pad_mask.sum(dim=1) - 1
            x = x[rows_ids, last_item_sessions]
            if self.multi_task>1:
                types1 = torch.Tensor.repeat(
                    self.eval_task, 
                    (x.shape[0],)
                ).cuda().long()
                x_emb1 = self.type_emb(types1)
                if self.use_tanh:
                    x_emb2 = torch.nn.Tanh()(x_emb1)
                    x1 = x*x_emb2
                    x = x1
                else:
                    x1 = torch.concat([x, x_emb1], axis=1)
                    x = self.mt_tower(x1)

        # Compute predictions probs
        x = self.pre(x)  # type: ignore

        return x

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor

    def calculate_metrics(self, predictions, targets) -> Dict[str, torch.Tensor]:  # type: ignore
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        outputs = {}
        predictions = self.forward_to_prediction_fn(predictions)

        for metric in self.metrics:
            outputs[self.metric_name(metric)] = metric(predictions, targets)

        return outputs

    def compute_metrics(self):
        metrics = {
            self.metric_name(metric): metric.compute()
            for metric in self.metrics
            if getattr(metric, "top_ks", None)
        }
        # Explode metrics for each cut-off
        # TODO make result generic:
        # To accept a mix of ranking metrics and others not requiring top_ks ?
        topks = {self.metric_name(metric): metric.top_ks for metric in self.metrics}
        results = {}
        for name, metric in metrics.items():
            for measure, k in zip(metric, topks[name]):
                results[f"{name}_{k}"] = measure
        return results


class CustomNextItemPredictionPrepareBlock(BuildableBlock):
    def __init__(
        self,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature

    def build(self, input_size) -> Block:
        return Block(
            _CustomNextItemPredictionTask(
                input_size,
                self.target_dim,
                self.weight_tying,
                self.item_embedding_table,
                self.softmax_temperature,
            ),
            [-1, self.target_dim],
        )


class _CustomNextItemPredictionTask(torch.nn.Module):
    """Predict the interacted item-id probabilities.
    - During inference, the task consists of predicting the next item.
    - During training, the class supports the following Language modeling tasks:
        Causal LM, Masked LM, Permutation LM and Replacement Token Detection
    Parameters:
    -----------
        input_size: int
            Input size of this module.
        target_dim: int
            Dimension of the target.
        weight_tying: bool
            The item id embedding table weights are shared with the prediction network layer.
        item_embedding_table: torch.nn.Module
            Module that's used to store the embedding table for the item.
        softmax_temperature: float
            Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
            Value 1.0 reduces to regular softmax.
    """

    def __init__(
        self,
        input_size: int,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        if self.weight_tying:
            self.output_layer_bias = torch.nn.Parameter(torch.Tensor(self.target_dim))
            torch.nn.init.zeros_(self.output_layer_bias)
        else:
            self.output_layer = torch.nn.Linear(
                self.input_size[-1], self.target_dim  # type: ignore
            )

    def forward(self, inputs) -> torch.Tensor:
        if isinstance(inputs, list):
            x = inputs[0]
            labels_all = inputs[1]
            item_seq = inputs[2]
            item_probs = inputs[3]
            item_correction = inputs[4]
            neg_factor = inputs[5]
            temperature = inputs[6]
            item_correction_factor = inputs[7]
            
            bs = x.shape[0]
            
            unique_labels = torch.unique(labels_all)
            if neg_factor>0:
                uni_neg = torch.randint(low=1, high=1825324, size=(int(neg_factor*bs),)).cuda()
                unique_labels = torch.unique(torch.cat([labels_all,uni_neg]))
            else:
                unique_labels = torch.unique(labels_all)
            
            emb_idx = self.item_embedding_table.weight[unique_labels]
            x = x @ emb_idx.t()
            item_probs_idx = item_probs[unique_labels]
            if item_correction:
                x = x - (item_correction_factor*torch.log(item_probs_idx+0.00000000001))
            
            # Masking Elements from Session
            if item_seq is not None:
                mask = []
                for i in range(bs):
                    mask.append(
                        torch.isin(unique_labels, item_seq[i])
                    )
                mask = torch.stack(mask)
                xx = torch.reshape(
                    torch.Tensor.repeat(unique_labels, bs),
                    (bs, -1)
                )
                # Keeping Current Label
                mask2 = (xx==torch.unsqueeze(labels_all, axis=1))
                mask[mask2] = False
                mask = ~mask
                x = x*mask
            
            pos = torch.unsqueeze(labels_all, axis=1)==unique_labels
            neg = ~pos
            x_pos = torch.unsqueeze(x[pos], axis=1)
            x_neg = torch.reshape(x[neg], (x_pos.shape[0], -1))

            x = torch.concat([x_pos, x_neg], axis=1)

            return x/temperature, torch.zeros_like(labels_all)
        else:
            x = inputs
            emb_idx = self.item_embedding_table.weight
            x = x @ emb_idx.t()
            return x

    def _get_name(self) -> str:
        return "NextItemPredictionTask"
    
from merlin_standard_lib import Registry, Schema, Tag

from transformers4rec.torch.utils.data_utils import MerlinDataLoader

import torch

device = 'cuda:0'

def to_sparse_tensor(values_offset, seq_limit):
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
    return _build_sparse_tensor(values, offsets, diff_offsets, num_rows, seq_limit)

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

def _get_indices(offsets, diff_offsets):
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

def _build_sparse_tensor(values, offsets, diff_offsets, num_rows, seq_limit):
    indices = _get_indices(offsets, diff_offsets)
    return _get_sparse_tensor(values, indices, num_rows, seq_limit)   

class MyCollator(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
    def __call__(self, batch):
        xb = batch[0][0]
        xc1 = to_sparse_tensor(xb['aid_'], self.max_seq_len)
        xc2 = to_sparse_tensor(xb['type_'], self.max_seq_len)
        return({'aid_': xc1, 'type_': xc2})

class CustomMerlinDataLoader(MerlinDataLoader):
    
        @classmethod
        def from_schema(
            cls,
            schema: Schema,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            continuous_features=None,
            categorical_features=None,
            targets=None,
            collate_fn=None,
            shuffle=True,
            buffer_size=0.06,
            parts_per_chunk=1,
            separate_labels=True,
            named_labels=False,
            sparse_names=None,
            sparse_max=None,
            **kwargs,
        ):
            """
               Instantitates `MerlinDataLoader` from a ``DatasetSchema``.
            Parameters
            ----------
                schema: DatasetSchema
                    Dataset schema
                paths_or_dataset: Union[str, Dataset]
                    Path to paquet data of Dataset object.
                batch_size: int
                    batch size of Dataloader.
                max_sequence_length: int
                    The maximum length of list features.
            """
            categorical_features = (
                categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
            )
            continuous_features = (
                continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
            )
            targets = targets or schema.select_by_tag(Tag.TARGETS).column_names
            schema = schema.select_by_name(categorical_features + continuous_features + targets)
            sparse_names = sparse_names or schema.select_by_tag(Tag.LIST).column_names
            sparse_max = sparse_max or {name: max_sequence_length for name in sparse_names}
            loader = cls(
                paths_or_dataset,
                batch_size=batch_size,
                max_sequence_length=max_sequence_length,
                labels=targets if separate_labels else [],
                cats=categorical_features if separate_labels else categorical_features + targets,
                conts=continuous_features,
                collate_fn=collate_fn,
                engine="parquet",
                shuffle=shuffle,
                buffer_size=buffer_size,  # how many batches to load at once
                parts_per_chunk=parts_per_chunk,
                schema=schema,
                **kwargs,
            )

            return loader
    

class CustomTrainer(Trainer):
    
    def set_shuffle(self, shuffle):
        self.shuffle=shuffle

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len=max_seq_len
    
    def get_train_dataloader(self):
#         from merlin.io import Dataset
#         from merlin.loader.torch import Loader
        if self.train_dataloader is not None:
            return self.train_dataloader

        assert self.schema is not None, "schema is required to generate Train Dataloader"

        # Set global_rank and global_size if DDP is used
        if self.args.local_rank != -1:
            local_rank = self.args.local_rank
            global_size = self.args.world_size
        else:
            local_rank = None
            global_size = None

        return CustomMerlinDataLoader.from_schema(
            self.schema,
            self.train_dataset_or_path,
            self.args.per_device_train_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=self.args.dataloader_drop_last,
            shuffle=self.shuffle,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
            global_rank=local_rank,
            global_size=global_size,
            collate_fn=MyCollator(max_seq_len=self.max_seq_len),
        )
        
    def get_test_dataloader(self, test_dataset=None):
        """
        Set the test dataloader to use by Trainer.
        It supports user defined data-loader set as an attribute in the constructor.
        When the attribute is None, The data-loader is defined using test_dataset
        and the `data_loader_engine` specified in Training Arguments.
        """
        if self.test_dataloader is not None:
            return self.test_dataloader

        if test_dataset is None and self.test_dataset_or_path is None:
            raise ValueError("Trainer: test requires an test_dataset.")
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset_or_path
        assert self.schema is not None, "schema is required to generate Test Dataloader"
        return CustomMerlinDataLoader.from_schema(
            self.schema,
            test_dataset,
            self.args.per_device_eval_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=False,
            shuffle=False,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
            collate_fn=MyCollator(max_seq_len=self.max_seq_len)
        )