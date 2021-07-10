# coding=utf8

from transformers import GPT2LMHeadModel, GPT2Config
import transformers
import torch
import os
from torch import nn


class Gpt2Generator(nn.Module):
    def __init__(self, conf):
        super(Gpt2Generator, self).__init__()
        self.conf = conf
        self.gpt_path = getattr(conf,"gpt_path",None)
        #TODO
        pretrain_name = 'gpt2'
        if self.gpt_path:
            # pretrain_name = self.gpt_path
            pretrain_name = os.path.join(self.gpt_path, 'config.json')
        # self.gpt = GPT2LMHeadModel.from_pretrained(pretrain_name)
        model_config = GPT2Config.from_json_file(pretrain_name)
        self.gpt = GPT2LMHeadModel(config=model_config)

    def forward(self, input_ids, labels):
        outputs = self.gpt(input_ids, labels = labels)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits
