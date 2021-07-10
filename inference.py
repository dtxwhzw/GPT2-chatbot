# coding=utf8

import torch
import os
from tqdm import tqdm
from datetime import datetime
from transformers import BertTokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from dataset import MyDataset, collator_fn
from utils.logger import create_logger
from utils.conf_utils import Config, parse_args
from torch.utils.data import dataloader, random_split
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from metrics import calculate_acc, calculate_loss
from model import Gpt2Generator
import numpy as np
import random
from utils.file_helper import get_real_path, mk_folder_for_file


class Inference:
    def __init__(self,args):
        self.conf = args
        self.logger = create_logger(get_real_path(self.conf.log_path))
        self.logger.info("Start Inference")
        self.gpt_path = getattr(self.conf, 'gpt_path', None)
        self.gpt_path = get_real_path(self.gpt_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.gpt_path)
        best_model_path = os.path.join(self.conf.model_path, 'best_acc_model.cpt')
        self.model = Gpt2Generator(self.conf)
        self.model = self.load_state_dict(best_model_path)
        self.device = self.conf.device
        if self.device != 'cuda' :
            self.logger.info("[Warning] Use cpu training!")
        self.cuda_ids = self.conf.device_ids
        self.logger.info(f'Using device: {self.device}')
        self.model.to(self.device)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    :param logits: logits distribution shape (vocab size)
    :param top_k: keep only top k tokens with highest probability (top-k filtering)
    :param top_p:  keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    :param filter_value:
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1)) # Safety check
    if top_k > 0:
        """
        Remove all tokens with a probability less than the last token of the top-k
        torch.topk()返回最后一维最大的top_k个元素，返回值为二维（values，indices）
        其他维度自行计算
        """
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][:,-1,None]
            logit[indices_to_remove] = filter_value # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1) # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threhold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:,:-1].clone()
        sorted_indices_to_remove[:,0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def main(args):
    infer = Inference(args)


if __name__ == '__main__' :
    import sys, json

    f_conf = parse_args(sys.argv[1])
    main(f_conf)