#!/usr/bin/env python
# coding:utf-8
import json
import copy
import torch
import random
import collections
from typing import List, Dict
from torch.utils.data import Dataset, Sampler


class RawDataLoaderBase:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self, path: str):
        pass


DataItem = collections.namedtuple("DataItem", ["text", "label", "wp_text", "wp_label", "wp_mark"])


class FewShotExample(object):
    def __init__(
            self,
            gid: int,
            batch_id: int,
            test_id: int,
            domain_name: str,
            support_data_items: List[DataItem],
            test_data_item: DataItem
    ):
        self.gid = gid
        self.batch_id = batch_id
        self.test_id = test_id  # test relative index in one batch
        self.domain_name = domain_name

        self.support_data_items = support_data_items  # all support data items
        self.test_data_item = test_data_item  # one test data items

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'gid:{}\n\tdomain:{}\n\ttest_data:{}\n\ttest_label:{}\n\tsupport_data:{}'.format(
            self.gid,
            self.domain_name,
            self.test_data_item.wp_text,
            self.test_data_item.wp_label,
            self.support_data_items,
        )


class FewShotRawDataLoader(RawDataLoaderBase):
    def __init__(self, debugging: bool=False):
        super(FewShotRawDataLoader, self).__init__()
        self.debugging = debugging
        self.idx_dict = {'O': 0, 'B': 1, 'I': 2}

    def load_data(self, path: str) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
            load few shot data set
            input:
                path: file path
            output
                examples: a list, all example loaded from path
                few_shot_batches: a list, of fewshot batch, each batch is a list of examples
                max_len: max sentence length
            """
        with open(path, 'r') as reader:
            raw_data = json.load(reader)
            examples, few_shot_batches, max_len, max_support_size, trans_mat = self.raw_data2examples(raw_data)
        if self.debugging:
            examples, few_shot_batches = examples[:8], few_shot_batches[:2]
        return examples, few_shot_batches, max_support_size, trans_mat

    def raw_data2examples(self, raw_data: Dict) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
        process raw_data into examples
        """
        examples = []
        all_len = []  # used to get max length
        all_support_size = []
        few_shot_batches = []
        # transition matrix
        trans_mat = torch.zeros(3, 5, dtype=torch.int32).tolist()
        start_trans_mat = torch.zeros(3, dtype=torch.int32).tolist()
        end_trans_mat = torch.zeros(3, dtype=torch.int32).tolist()
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                one_batch_examples = []
                support_data_items, test_data_items = self.batch2data_items(batch)
                # update transition matrix
                self.get_trans_mat(trans_mat, start_trans_mat, end_trans_mat, support_data_items)

                all_support_size.append(len(support_data_items))
                ''' Pair each test sample with full support set '''
                for test_id, test_data_item in enumerate(test_data_items):
                    gid = len(examples)
                    example = FewShotExample(
                        gid=gid,
                        batch_id=batch_id,
                        test_id=test_id,
                        domain_name=domain_n,
                        test_data_item=test_data_item,
                        support_data_items=support_data_items,
                    )
                    examples.append(example)
                    one_batch_examples.append(example)
                all_len.extend([len(l.wp_text) for l in test_data_items] + [len(l.wp_text) for l in support_data_items])
                few_shot_batches.append(one_batch_examples)
        max_len = max(all_len)
        max_support_size = max(all_support_size)
        return examples, few_shot_batches, max_len, max_support_size, (trans_mat, start_trans_mat, end_trans_mat)

    def get_data_items(self, parts: dict) -> List[DataItem]:
        data_item_lst = []
        for text, label, wp_text, wp_label, wp_mark in zip(
                parts['seq_ins'], parts['seq_outs'],
                parts['tokenized_texts'], parts['word_piece_labels'], parts['word_piece_marks']):
            data_item = DataItem(text=text, label=label, wp_text=wp_text, wp_label=wp_label, wp_mark=wp_mark)
            data_item_lst.append(data_item)
        return data_item_lst

    def batch2data_items(self, batch: dict) -> (List[DataItem], List[DataItem]):
        support_data_items = self.get_data_items(parts=batch['support'])
        test_data_items = self.get_data_items(parts=batch['batch'])
        return support_data_items, test_data_items

    def get_trans_mat(self,
                      trans_mat: List[List[int]],
                      start_trans_mat: List[int],
                      end_trans_mat: List[int],
                      support_data: List[str]) -> None:
        for support_data_item in support_data:
            labels = support_data_item.label
            s_idx = self.idx_dict[labels[0][0]]
            e_idx = self.idx_dict[labels[-1][0]]
            start_trans_mat[s_idx] += 1
            end_trans_mat[e_idx] += 1
            for i in range(len(labels) - 1):
                cur_label = labels[i]
                next_label = labels[i + 1]
                start_idx = self.idx_dict[cur_label[0]]
                if cur_label == next_label:
                    end_idx = self.idx_dict[next_label[0]]
                else:
                    if cur_label[0] == next_label[0]:
                        end_idx = self.idx_dict[next_label[0]] + 2
                    else:
                        if cur_label == 'O':
                            end_idx = self.idx_dict[next_label[0]]
                        elif next_label == 'O':
                            end_idx = 0
                        else:
                            end_idx = self.idx_dict[next_label[0]] \
                                if cur_label[2:] == next_label[2:] else self.idx_dict[next_label[0]] + 2

                trans_mat[start_idx][end_idx] += 1


class FewShotDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, tensor_features: List[List[torch.Tensor]]):
        self.tensor_features = tensor_features

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        return self.tensor_features[index]

    def __len__(self):
        return len(self.tensor_features)


def pad_tensor(vec: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    if pad - vec.size(dim) == 0:  # reach max no need to pad (This is used to avoid pytorch0.4.1's bug)
        return vec
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    ret = torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)
    return ret


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in a batch of sequences.
    Notice:
        - Use 0 as pad value.
        - Support 2 pad position at same time with optional args.
        - Batch format slight different from "TensorDataset"
        (where batch is (Tensor1, Tensor2, ..., Tensor_n), and shape of Tensor_i is (batch_size, item_i_size),
        Here, batch format is List[List[torch.Tensor]], it's shape is (batch_size, item_num, item_size).
        This is better for padding at running, and write get_item() for dataset.
    """

    def __init__(self, dim: int = 0, sp_dim: int = None, sp_item_idx: List[int] = None):
        """
        args:
            dim - the dimension to be padded
            sp_dim - the dimension to be padded for some special item in batch(this leaves some flexibility for pad dim)
            sp_item_idx - the index for some special item in batch(this leaves some flexibility for pad dim)
        """
        self.dim = dim
        self.sp_dim = sp_dim
        self.sp_item_idx = sp_item_idx

    def pad_collate(self, batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        args:
            batch - list of (tensor1, tensor2, ..., tensor3)

        reutrn:
            ret - tensors of all examples in 'batch' after padding,
            each tensor belongs to one items type.
        """
        ret = []
        for item_idx in range(len(batch[0])):  # pad each data item
            # find longest sequence
            max_len = max(map(lambda x: x[item_idx].shape[self.get_dim(item_idx)], batch))
            # pad according to max_len
            padded_item_lst = list(map(
                lambda x: pad_tensor(x[item_idx], pad=max_len, dim=self.get_dim(item_idx)), batch))
            # stack all
            padded_item_lst = torch.stack(padded_item_lst, dim=0)
            ret.append(padded_item_lst)
        return ret

    def get_dim(self, item_idx):
        """ this dirty function is design for bert non-word-piece index.
        This will be removed by move index construction to BertContextEmbedder
        """
        if self.sp_dim and self.sp_item_idx and item_idx in self.sp_item_idx:
            return self.sp_dim  # pad to the special dimension
        else:
            return self.dim  # pad to the dimension

    def __call__(self, batch):
        return self.pad_collate(batch)


class SimilarLengthSampler(Sampler):
    r"""
    Samples elements and ensure
        1. each batch element has similar length to reduce padding.
        2. each batch is in decent length order (useful to pack_sequence for RNN)
        3. batches are ordered randomly
    If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        batch_size (int): num of samples in one batch
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        all_idxs = list(range(len(data_source)))
        all_lens = [self.get_length(idx) for idx in all_idxs]
        self.all_index = self.sort_and_batching(all_idxs, all_lens, batch_size)
        super(SimilarLengthSampler, self).__init__()

    def sort_and_batching(self, all_idxs, all_lens, batch_size):
        sorted_idxs = sorted(zip(all_idxs, all_lens), key=lambda x: x[1], reverse=True)
        sorted_idxs = [item[0] for item in sorted_idxs]
        batches = self.chunk(sorted_idxs, batch_size)  # shape: (batch_num, batch_size)
        random.shuffle(batches)  # shuffle batches
        flatten_batches = collections._chain.from_iterable(batches)
        return flatten_batches

    def chunk(self, lst, n):
        return [lst[i: i + n] for i in range(0, len(lst), n)]

    def get_length(self, idx):
        return len(self.data_source[idx][0])  # we use the test length in sorting

    def __iter__(self):
        return iter(copy.deepcopy(self.all_index))  # if not deep copy, iteration will stop after first step

    def __len__(self):
        return len(self.data_source)
