# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self,
                 base_model,
                 tokenizer,
                 out_dim,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False):
        super().__init__()
        self.config = base_model.config
        
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            # self.v_head = nn.Linear(self.config.word_embed_proj_dim,
            #                         1,
            #                         bias=False)
            self.linear=nn.Linear(in_features=self.config.word_embed_proj_dim,out_features=out_dim)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            # self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self.linear=nn.Linear(in_features=self.config.n_embd,out_features=out_dim)

        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                has_negatives=False,
                use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        end_id=torch.sum(attention_mask,dim=-1)-1
        
        hidden_states = transformer_outputs[0][range(len(end_id)),end_id,:]
        out_features=self.linear(hidden_states)
        return {
            "out_features":out_features
        }
