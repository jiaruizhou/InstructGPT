# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from dschat.utils.model.reward_model import RewardModel
from dschat.utils.utils import load_state_dict_into_model, print_rank_0


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def causal_lm_model_to_fp32_loss(model):
    """ Convert CausalLM model to calculate loss in fp32 """

    def causal_lm_forward(
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ):
        kwargs = dict() if model.config.model_type == "llama" else dict(
            head_mask=head_mask)
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length))

        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss, ) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = causal_lm_forward


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    # why? 
    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        dropout=None,
                        out_dim=128,
                        zero_stage=0,
                        compute_fp32_loss=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, dropout)
    end = time.time()
    print_rank_0(f">Creating model from_config took {end - start} seconds",
                 None)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        out_dim,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss)

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)
        
        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()

        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)

    return critic_model


class InfoNCE(torch.nn.Module):
    def __init__(self) -> None:
        super(InfoNCE,self).__init__()
    def forward(self,queries,positives):
        bs=queries.shape[0]
        sim_matrix = torch.mm(queries, positives.t())
        label = torch.arange(bs, device=sim_matrix.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix, label)
        return loss


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.95):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, query,passage):
        sim=torch.mm(query,passage.t())

        label = torch.eye(query.shape[0]).float().to(query.device)
        distances = torch.mean(torch.cdist(query.unsqueeze(0).repeat(query.shape[0], 1, 1).float(), \
            passage.unsqueeze(1).repeat(1, passage.shape[0], 1).float(), p=2), dim=-1)
        whole_loss = ((label) * torch.pow(distances, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
        
        loss_contrastive = torch.mean(whole_loss)
        # import pdb
        # pdb.set_trace()
        return loss_contrastive

