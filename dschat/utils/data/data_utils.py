# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
import hashlib
from itertools import chain
from dschat.utils.data import raw_datasets
from deepspeed.accelerator import get_accelerator


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
                                                 dataset_name, chat_path)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank,
                                output_path,
                                dataset_name,
                                seed,
                                split_name,
                                data_split,
                                split_index,
                                data_size,
                                rebuild=False):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name
                                                            == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 3:
        filtered = 0
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                    for key_word in ["input_ids", "attention_mask"]:
                        prompt_token[key_word] = prompt_token[
                            key_word].squeeze(0).flip(0)
                    prompt_dataset.append(prompt_token)
                else:
                    filtered += 1
        print(f'Creating dataset {raw_dataset.dataset_name_clean} '
              f'for {train_phase=} size={len(prompt_dataset)} {filtered=}')

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset), rebuild)
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset), rebuild)
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f'Creating prompt dataset {data_path}, {reload=}')
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank,
                data_path[0],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank,
                    d_path,
                    data_split,
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload)
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)


class MSMARCODataset:

    def __init__(self, data_path, seed=None, local_rank=-1):
        self.data_path = data_path
        self.seed = seed
        self.local_rank = local_rank

    @property
    def docid2doc(self):
        if not hasattr(self, "_docid2doc"):
            with open(os.path.join(self.data_path, "collection.tsv"), "r") as f:
                self._docid2doc = dict([line.strip().split("\t") for line in tqdm(f, desc="loading collection")])
        return self._docid2doc

    @property
    def train_qid2query(self):
        if not hasattr(self, "_train_qid2query"):
            with open(os.path.join(self.data_path, "queries.train.tsv"), "r") as f:
                self._train_qid2query = dict([line.strip().split("\t") for line in tqdm(f, desc="loading train queries")])
        return self._train_qid2query

    @property
    def eval_qid2query(self):
        if not hasattr(self, "_eval_qid2query"):
            with open(os.path.join(self.data_path, "queries.dev.small.tsv"), "r") as f:
                self._eval_qid2query = dict([line.strip().split("\t") for line in tqdm(f, desc="loading eval queries")])
        return self._eval_qid2query

    @property
    def train_qid2docids(self):
        if not hasattr(self, "_train_qid2docids"):
            self._train_qid2docids = {}
            with open(os.path.join(self.data_path, "qrels.train.tsv"), "r") as f:
                for line in tqdm(f, desc="loading train qrels"):
                    qid, _, docid, _ = line.strip().split("\t")
                    self._train_qid2docids.setdefault(qid, []).append(docid)
        return self._train_qid2docids
    
    @property
    def eval_qid2docids(self):
        if not hasattr(self, "_eval_qid2docids"):
            self._eval_qid2docids = {}
            with open(os.path.join(self.data_path, "qrels.dev.small.tsv"), "r") as f:
                for line in tqdm(f, desc="loading eval qrels"):
                    qid, _, docid, _ = line.strip().split("\t")
                    self._eval_qid2docids.setdefault(qid, []).append(docid)
        return self._eval_qid2docids

    @property
    def train_text_pairs(self):
        if not hasattr(self, "_train_text_pairs"):
            self._train_text_pairs = []
            for qid, docids in self.train_qid2docids.items():
                for docid in docids:
                    self._train_text_pairs.append((self.train_qid2query[qid], self.docid2doc[docid]))
        return self._train_text_pairs

    @property
    def eval_text_pairs(self):
        if not hasattr(self, "_eval_text_pairs"):
            self._eval_text_pairs = []
            for qid, docids in self.eval_qid2docids.items():
                for docid in docids:
                    self._eval_text_pairs.append((self.eval_qid2query[qid], self.docid2doc[docid]))
        return self._eval_text_pairs

    def get_train_data(self):
        return self.train_text_pairs
    
    def get_eval_data(self):
        return self.eval_text_pairs


class RetrievalDataset(Dataset):

    def __init__(self, query_dataset, positive_dataset, negative_dataset) -> None:
        super().__init__()
        assert len(query_dataset) == len(positive_dataset) == len(negative_dataset), \
            "The length of query_dataset, positive_dataset, and negative_dataset should be the same."
        self.query_dataset = query_dataset
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset

    def __len__(self):
        return len(self.query_dataset)

    def __getitem__(self, idx):
        return self.query_dataset[idx]["input_ids"], self.query_dataset[idx]["attention_mask"], \
            self.positive_dataset[idx]["input_ids"], self.positive_dataset[idx]["attention_mask"], \
            self.negative_dataset[idx]["input_ids"], self.negative_dataset[idx]["attention_mask"]


def tokenize_raw_dataset(raw_dataset, tokenizer, end_of_text_token, max_seq_len):
    query_dataset = []
    positive_dataset = []
    negative_dataset = []

    # If negatives are not provided, we will create a dataset with None values
    if len(raw_dataset) > 0 and len(raw_dataset[0]) == 2:
        raw_dataset = [(query, positive, None) for query, positive in raw_dataset]

    for i, (query, positive, negative) in enumerate(tqdm(raw_dataset, desc="tokenizing dataset")):
        query += end_of_text_token
        query_token = tokenizer(query,
                                max_length=max_seq_len,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
        query_dataset.append(query_token)

        positive += end_of_text_token
        positive_token = tokenizer(positive,
                                 max_length=max_seq_len,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
        positive_dataset.append(positive_token)

        if negative is not None:
            negative += end_of_text_token
            negative_token = tokenizer(negative,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            negative_dataset.append(negative_token)
        else:
            negative_dataset.append({"input_ids": None, "attention_mask": None})

    return RetrievalDataset(query_dataset, positive_dataset, negative_dataset)


def create_retrieval_dataset(local_rank,
                             data_path,
                             cache_path,
                             seed,
                             tokenizer,
                             max_seq_len,
                             end_of_text_token="<|endoftext|>",
                             reload=False):
    """
    Creates the retrieval dataset
    """
    assert len(data_path) == 1, "Retrieval dataset does not support multiple datasets."
    data_path = data_path[0]

    os.makedirs(cache_path, exist_ok=True)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{data_path}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{cache_path}/traindata_{fname}.pt"
    eval_fname = f"{cache_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        buf_create_cache = torch.ByteTensor([not cache_found]).to(
            get_accelerator().current_device_name())
        torch.distributed.all_reduce(buf_create_cache)
    else:
        buf_create_cache = torch.ByteTensor([not cache_found])

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f'Creating retrieval dataset {data_path}, {reload=}')

        raw_dataset = MSMARCODataset(data_path, seed, local_rank)

        train_dataset = raw_dataset.get_train_data()
        train_dataset = tokenize_raw_dataset(train_dataset, tokenizer, end_of_text_token, max_seq_len)

        eval_dataset = raw_dataset.get_eval_data()
        eval_dataset = tokenize_raw_dataset(eval_dataset, tokenizer, end_of_text_token, max_seq_len)

        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorRetrieval:

    def __call__(self, data):
        batch = {}
        if data[0][-2] is None:
            batch["has_negatives"] = False
            batch["input_ids"] = torch.cat([f[0]
                                            for f in data] + [f[2] for f in data],
                                        dim=0)
            batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                                [f[3] for f in data],
                                                dim=0)
        else:
            batch["has_negatives"] = True
            batch["input_ids"] = torch.cat([f[0]
                                            for f in data] + [f[2] for f in data] + [f[4] for f in data],
                                        dim=0)
            batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                                [f[3] for f in data] + [f[5] for f in data],
                                                dim=0)
        return batch


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []

# # Copyright (c) Microsoft Corporation.
# # SPDX-License-Identifier: Apache-2.0

# # DeepSpeed Team
# """
# Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
# """
# import torch
# from torch.utils.data import Dataset, Subset, ConcatDataset
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn.functional as F
# from datasets import load_dataset
# import numpy as np
# import os
# import hashlib
# from itertools import chain
# from dschat.utils.data import raw_datasets
# from deepspeed.accelerator import get_accelerator


# def get_raw_dataset(dataset_name, output_path, seed, local_rank):
#     if "unicamp-dl/mmarco" in dataset_name:
#         return raw_datasets.mmarcoDataset(output_path, seed,local_rank, dataset_name)
#     elif "Dahoas/rm-static" in dataset_name:
#         return raw_datasets.DahoasRmstaticDataset(output_path, seed,
#                                                   local_rank, dataset_name)
#     elif "Dahoas/full-hh-rlhf" in dataset_name:
#         return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
#                                                     local_rank, dataset_name)
#     elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
#         return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "yitingxie/rlhf-reward-datasets" in dataset_name:
#         return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "openai/webgpt_comparisons" in dataset_name:
#         return raw_datasets.OpenaiWebgptcomparisonsDataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "stanfordnlp/SHP" in dataset_name:
#         return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
#                                                   local_rank, dataset_name)
#     elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
#         return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "wangrui6/Zhihu-KOL" in dataset_name:
#         return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
#                                                     local_rank, dataset_name)
#     elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
#         return raw_datasets.CohereMiraclzhqueries2212Dataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
#         return raw_datasets.HelloSimpleAIHC3ChineseDataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "mkqa-Chinese" in dataset_name:
#         return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
#                                                "mkqa")
#     elif "mkqa-Japanese" in dataset_name:
#         return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
#                                                 "mkqa")
#     elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
#         return raw_datasets.CohereMiracljaqueries2212Dataset(
#             output_path, seed, local_rank, dataset_name)
#     elif "lmqg/qg_jaquad" in dataset_name:
#         return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
#                                                 dataset_name)
#     elif "lmqg/qag_jaquad" in dataset_name:
#         return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
#                                                  dataset_name)
#     elif "local/jsonfile" in dataset_name:
#         chat_path = os.path.abspath(
#             os.path.join(os.path.dirname(__file__), os.path.pardir,
#                          os.path.pardir, os.path.pardir))
#         if not (os.path.isfile(chat_path + '/data/train.json')
#                 and os.path.isfile(chat_path + '/data/eval.json')):
#             raise RuntimeError(
#                 f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
#             )
#         return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
#                                                  dataset_name, chat_path)
#     else:
#         raise RuntimeError(
#             f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
#         )


# def get_shuffle_idx(seed, size):
#     np_rng = np.random.RandomState(seed=seed)
#     dtype_ = np.uint32
#     if size >= (np.iinfo(np.uint32).max - 1):
#         dtype_ = np.int64
#     shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
#     np_rng.shuffle(shuffle_idx)
#     return shuffle_idx


# def get_raw_dataset_split_index(local_rank,
#                                 output_path,
#                                 dataset_name,
#                                 seed,
#                                 split_name,
#                                 data_split,
#                                 split_index,
#                                 data_size,
#                                 rebuild=False):
#     index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
#     # reindex each time when using local jsonfile since it's more likely to get modified
#     if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name
#                                                             == 'jsonfile'):
#         splits = [float(s) for s in data_split.split(',')]
#         splits_sum = sum(splits)
#         splits = [split / splits_sum for split in splits]
#         splits_index = [0]
#         for index, split in enumerate(splits):
#             splits_index.append(splits_index[index] +
#                                 int(round(split * float(data_size))))
#         diff = splits_index[-1] - data_size
#         for index in range(1, len(splits_index)):
#             splits_index[index] -= diff
#         assert splits_index[-1] == data_size

#         shuffle_idx = get_shuffle_idx(seed, data_size)
#         for split_i in range(len(splits)):
#             shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
#             shuffle_idx_split = shuffle_idx[
#                 splits_index[split_i]:splits_index[split_i + 1]]
#             np.save(shuffle_idx_split_file_name,
#                     shuffle_idx_split,
#                     allow_pickle=True)
#     index = np.load(index_file_name, allow_pickle=True)
#     return index.tolist()


# class PromptDataset(Dataset):

#     def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
#                  pad_token_id, train_phase) -> None:
#         super().__init__()
#         self.prompt_dataset = prompt_dataset
#         self.chosen_dataset = chosen_dataset
#         self.reject_dataset = reject_dataset
#         self.pad_token_id = pad_token_id
#         self.train_phase = train_phase

#     def __len__(self):
#         length = len(self.chosen_dataset)
#         if self.train_phase == 3:
#             length = len(self.prompt_dataset)
#         return length

#     def __getitem__(self, idx):
#         if self.train_phase == 1:
#             return {
#                 "input_ids": self.chosen_dataset[idx]["input_ids"],
#                 "attention_mask": self.chosen_dataset[idx]["attention_mask"],
#                 "labels": self.chosen_dataset[idx]["input_ids"]
#             }
#         elif self.train_phase == 2:
#             return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
#                 self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
#         elif self.train_phase == 3:
#             return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
#                 self.pad_token_id


# def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
#                          end_of_conversation_token, max_seq_len):
#     prompt_dataset = []
#     chosen_dataset = []
#     reject_dataset = []
#     query_dataset = []
#     passage_dataset = []
#     if train_phase == 1:
#         for i, tmp_data in enumerate(current_dataset):
#             # tokenize the text
#             chosen_sentence = raw_dataset.get_prompt_and_chosen(
#                 tmp_data)  # the accept response
#             if chosen_sentence is not None:
#                 chosen_sentence += end_of_conversation_token
#                 chosen_token = tokenizer(chosen_sentence,
#                                          max_length=max_seq_len,
#                                          padding="max_length",
#                                          truncation=True,
#                                          return_tensors="pt")
#                 chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
#                     0)
#                 chosen_token["attention_mask"] = chosen_token[
#                     "attention_mask"].squeeze(0)
#                 chosen_dataset.append(chosen_token)
#         print(
#             f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
#         )

#     elif train_phase == 2:
#         for i, tmp_data in enumerate(current_dataset):
#             # tokenize the text
#             passage=raw_datasets.get_passage(tmp_data)
#             query=raw_datasets.get_query(tmp_data)

#             passage_token= tokenizer(passage,
#                                         max_length=max_seq_len,
#                                         padding="max_length",
#                                         truncation=True,
#                                         return_tensors="pt")
#             query_token= tokenizer(query,
#                                         max_length=max_seq_len,
#                                         padding="max_length",
#                                         truncation=True,
#                                         return_tensors="pt")
#             query_token["input_ids"] = query_token["input_ids"]
#             query_token["attention_mask"] = query_token["attention_mask"]
#             query_dataset.append(query_token)


#             passage_token["input_ids"] = passage_token["input_ids"]
#             passage_token["attention_mask"] = passage_token["attention_mask"]
#             passage_dataset.append(passage_token)

#             print(
#                 f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(query_dataset)}'
#             )
            
#             return PromptDataset(prompt_dataset, query_dataset, passage_dataset,
#                                 tokenizer.pad_token_id, train_phase)
#         #     chosen_sentence = raw_dataset.get_prompt_and_chosen(
#         #         tmp_data)  # the accept response
#         #     reject_sentence = raw_dataset.get_prompt_and_rejected(
#         #         tmp_data)  # the accept response
#         #     if chosen_sentence is not None and reject_sentence is not None:
#         #         chosen_sentence += end_of_conversation_token  # the accept response
#         #         reject_sentence += end_of_conversation_token
#         #         chosen_token = tokenizer(chosen_sentence,
#         #                                  max_length=max_seq_len,
#         #                                  padding="max_length",
#         #                                  truncation=True,
#         #                                  return_tensors="pt")
#         #         reject_token = tokenizer(reject_sentence,
#         #                                  max_length=max_seq_len,
#         #                                  padding="max_length",
#         #                                  truncation=True,
#         #                                  return_tensors="pt")
#         #         chosen_token["input_ids"] = chosen_token["input_ids"]
#         #         chosen_token["attention_mask"] = chosen_token["attention_mask"]
#         #         chosen_dataset.append(chosen_token)

#         #         reject_token["input_ids"] = reject_token["input_ids"]
#         #         reject_token["attention_mask"] = reject_token["attention_mask"]
#         #         reject_dataset.append(reject_token)
#         # print(
#         #     f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
#         # )

#     elif train_phase == 3:
#         filtered = 0
#         for i, tmp_data in enumerate(current_dataset):
#             # tokenize the text
#             prompt = raw_dataset.get_prompt(tmp_data)
#             if prompt is not None:
#                 prompt_token = tokenizer(prompt, return_tensors="pt")
#                 if prompt_token["input_ids"].size()[-1] <= max_seq_len:
#                     for key_word in ["input_ids", "attention_mask"]:
#                         prompt_token[key_word] = prompt_token[
#                             key_word].squeeze(0).flip(0)
#                     prompt_dataset.append(prompt_token)
#                 else:
#                     filtered += 1
#         print(f'Creating dataset {raw_dataset.dataset_name_clean} '
#               f'for {train_phase=} size={len(prompt_dataset)} {filtered=}')

#     return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
#                          tokenizer.pad_token_id, train_phase)


# def create_dataset(local_rank, dataset_name, data_split, output_path,
#                    train_phase, seed, tokenizer, end_of_conversation_token,
#                    max_seq_len, rebuild):
#     raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
#     train_dataset = raw_dataset.get_train_data()
#     # train_index = get_raw_dataset_split_index(local_rank, output_path,
#     #                                           raw_dataset.dataset_name_clean,
#     #                                           seed, "train", data_split,
#     #                                           train_phase - 1,
#     #                                           len(train_dataset), rebuild)
#     # train_dataset = Subset(train_dataset, train_index)
#     train_dataset = create_dataset_split(train_dataset, raw_dataset,
#                                          train_phase, tokenizer,
#                                          end_of_conversation_token,
#                                          max_seq_len)

#     eval_dataset = raw_dataset.get_eval_data()
#     # eval_index = get_raw_dataset_split_index(local_rank, output_path,
#     #                                          raw_dataset.dataset_name_clean,
#     #                                          seed, "eval",
#     #                                          data_split, train_phase - 1,
#     #                                          len(eval_dataset), rebuild)
#     # eval_dataset = Subset(eval_dataset, eval_index)
#     eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
#                                         tokenizer, end_of_conversation_token,
#                                         max_seq_len)
#     return train_dataset, eval_dataset

# # def create_passage_dataset():

# def create_prompt_dataset(local_rank,
#                           data_path,
#                           data_split,
#                           output_path,
#                           train_phase,
#                           seed,
#                           tokenizer,
#                           max_seq_len,
#                           end_of_conversation_token="<|endoftext|>",
#                           sft_only_data_path=[],
#                           reload=False):
#     """
#     Creates the prompt dataset
#     """
#     os.makedirs(output_path, exist_ok=True)
#     fname = "_".join(data_path)
#     sft_cache_key = "_".join(sft_only_data_path)
#     tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
#     fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
#     fname = "_".join(fname.split("/"))
#     fname = hashlib.sha256(fname.encode()).hexdigest(
#     )  # hash the file name to avoid too long file name
#     train_fname = f"{output_path}/traindata_{fname}.pt"
#     eval_fname = f"{output_path}/evaldata_{fname}.pt"

#     cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
#     buf_create_cache = torch.ByteTensor([not cache_found]).to(
#         get_accelerator().current_device_name())
#     torch.distributed.all_reduce(buf_create_cache)

#     if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
#         print(f'Creating prompt dataset {data_path}, {reload=}')
#         if len(data_path) == 1:  # Single dataset.
#             train_dataset, eval_dataset = create_dataset(
#                 local_rank,
#                 data_path[0],
#                 data_split,
#                 output_path,
#                 train_phase,
#                 seed,
#                 tokenizer,
#                 end_of_conversation_token,
#                 max_seq_len,
#                 rebuild=reload)
#         else:  # Blending datasets.
#             train_datasets = []
#             eval_datasets = []
#             train_size = 0
#             eval_size = 0
#             for d_path in data_path:
#                 train_dataset, eval_dataset = create_dataset(
#                     local_rank,
#                     d_path,
#                     data_split,
#                     output_path,
#                     train_phase,
#                     seed,
#                     tokenizer,
#                     end_of_conversation_token,
#                     max_seq_len,
#                     rebuild=reload)
#                 train_datasets.append(train_dataset)
#                 eval_datasets.append(eval_dataset)
#                 train_size += len(train_dataset)
#                 eval_size += len(eval_dataset)
#             train_dataset = ConcatDataset(train_datasets)
#             shuffle_idx = get_shuffle_idx(seed, train_size)
#             train_dataset = Subset(train_dataset, shuffle_idx.tolist())
#             eval_dataset = ConcatDataset(eval_datasets)
#             shuffle_idx = get_shuffle_idx(seed, eval_size)
#             eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

#         # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
#         if train_phase == 1 and sft_only_data_path:
#             sft_train_datasets = []
#             sft_eval_datasets = []
#             sft_train_size = 0
#             sft_eval_size = 0
#             for sft_path in sft_only_data_path:
#                 sft_train_dataset, sft_eval_dataset = create_dataset(
#                     local_rank,
#                     sft_path,
#                     "10,0,0",
#                     output_path,
#                     train_phase,
#                     seed,
#                     tokenizer,
#                     end_of_conversation_token,
#                     max_seq_len,
#                     rebuild=reload)
#                 sft_train_datasets.append(sft_train_dataset)
#                 sft_eval_datasets.append(sft_eval_dataset)
#                 sft_train_size += len(sft_train_dataset)
#                 sft_eval_size += len(sft_eval_dataset)
#             if sft_train_datasets:  # Check if sft_train_datasets is not empty
#                 sft_train_dataset = ConcatDataset(sft_train_datasets)
#                 train_dataset = ConcatDataset(
#                     [train_dataset, sft_train_dataset])
#                 shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
#                 train_dataset = Subset(train_dataset, shuffle_idx.tolist())
#             if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
#                 sft_eval_dataset = ConcatDataset(sft_eval_datasets)
#                 eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
#                 shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
#                 eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
#         torch.save(train_dataset, train_fname)
#         torch.save(eval_dataset, eval_fname)
#     torch.distributed.barrier()
#     return torch.load(train_fname), torch.load(eval_fname)


# class DataCollatorReward:

#     def __call__(self, data):
#         batch = {}
#         batch["input_ids"] = torch.cat([f[0]
#                                         for f in data] + [f[2] for f in data],
#                                        dim=0)
#         batch["attention_mask"] = torch.cat([f[1] for f in data] +
#                                             [f[3] for f in data],
#                                             dim=0)
#         return batch


# class DataCollatorRLHF:

#     def __init__(self, max_token_len, inference_tp_size):
#         self.max_token_len = max_token_len
#         self.inference_tp_size = inference_tp_size

#     def __call__(self, data):
#         batch = {}
#         pad_token_id = data[-1][-1]

#         prompt = pad_sequence([f[0] for f in data],
#                               padding_value=pad_token_id,
#                               batch_first=True)
#         prompt_mask = pad_sequence([f[1] for f in data],
#                                    padding_value=0,
#                                    batch_first=True)

#         ### make sure the final ouput is a seqence of 2**?
#         length = prompt.size()[-1]
#         pad_length = self.max_token_len - length
#         if pad_length > 0:
#             batch["prompt"] = F.pad(prompt,
#                                     pad=(0, pad_length),
#                                     mode='constant',
#                                     value=pad_token_id)
#             batch["prompt_att_mask"] = F.pad(prompt_mask,
#                                              pad=(0, pad_length),
#                                              mode='constant',
#                                              value=0)
#         else:
#             batch["prompt"] = prompt
#             batch["prompt_att_mask"] = prompt_mask
#         batch["prompt"] = batch["prompt"].flip(1)
#         batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
#         return batch


# def get_unsupervised_data(args, tokenizer):
#     unsupervised_raw_datasets = load_dataset(
#         args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
#     column_names = unsupervised_raw_datasets["train"].column_names
#     text_column_name = "text" if "text" in column_names else column_names[0]

#     def tokenize_function(examples):
#         return tokenizer(examples[text_column_name])

#     tokenized_datasets = unsupervised_raw_datasets.map(
#         tokenize_function,
#         batched=True,
#         num_proc=args.preprocessing_num_workers,
#         remove_columns=column_names,
#         load_from_cache_file=True,
#         desc="Running tokenizer on dataset",
#     )

#     block_size = args.max_prompt_seq_len + args.max_answer_seq_len

#     def group_texts(examples):
#         # Concatenate all texts.
#         concatenated_examples = {
#             k: list(chain(*examples[k]))
#             for k in examples.keys()
#         }
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#         if total_length >= block_size:
#             total_length = (total_length // block_size) * block_size
#         # Split by chunks of max_len.
#         result = {
#             k:
#             [t[i:i + block_size] for i in range(0, total_length, block_size)]
#             for k, t in concatenated_examples.items()
#         }
#         result["labels"] = result["input_ids"].copy()
#         return result

#     lm_datasets = tokenized_datasets.map(
#         group_texts,
#         batched=True,
#         num_proc=args.preprocessing_num_workers,
#         load_from_cache_file=True,
#         desc=f"Grouping texts in chunks of {block_size}",
#     )

#     train_dataset = lm_datasets["train"]

#     return train_dataset


# class MiniDataset:

#     def __init__(self, max_size, small_batch_size):
#         self.dataset = []
#         self.max_size = max_size
#         self.small_batch_size = small_batch_size

#     def seperate(self):
#         small_dataset = []
#         for large_batch in self.dataset:
#             if type(large_batch) == list or type(large_batch) == tuple:
#                 large_size = len(large_batch[0])
#             elif type(large_batch) == dict:
#                 large_size = len(large_batch[list(large_batch.keys())[0]])
#             else:
#                 large_size = len(large_batch)
#             for i in range(0, large_size, self.small_batch_size):
#                 if type(large_batch) == list or type(large_batch) == tuple:
#                     small_dataset.append(
#                         [x[i:i + self.small_batch_size] for x in large_batch])
#                 elif type(large_batch) == dict:
#                     small_dataset.append({
#                         k: v[i:i + self.small_batch_size]
#                         for k, v in large_batch.items()
#                     })
#                 else:
#                     small_dataset.append(large_batch[i:i +
#                                                      self.small_batch_size])
#         self.free()

#         return small_dataset

#     def add(self, data):
#         if len(self.dataset) < self.max_size:
#             self.dataset.append(data)
#             if len(self.dataset) == self.max_size:
#                 return self.seperate()
#             else:
#                 return None
#         else:
#             raise ValueError(
#                 "The dataset is full but we did not stop it. There is a bug in the code."
#             )

#     def free(self):
#         self.dataset = []
