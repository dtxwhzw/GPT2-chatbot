# coding=utf8

import torch
from torch.utils.data.dataset import Dataset
from utils.logger import create_logger
from tqdm import tqdm
import numpy as np
from utils.file_helper import is_file_exists
import pickle


class MyDataset(Dataset) :
    START_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    PAD_TOKEN = "[PAD]"

    def __init__(self, tokenizer, data_path, logger, max_len, use_mmi: bool) :
        super(MyDataset, self).__init__()
        self.logger = logger
        self.tokenizer = tokenizer
        self.data_path = data_path
        with open(self.data_path, 'rb') as f :
            self.data = f.read().decode('utf-8')
        # linux 和 windows环境下的空格的换行符不一样
        if "\r\n" in self.data :
            self.data = self.data.split('\r\n\r\n')
        else :
            self.data = self.data.split('\n\n')
        self.logger.info("loading the data, there are {} dialogue in dataset.".format(
            len(self.data)
        ))
        self.max_len = max_len
        self.mmi = use_mmi
        if self.mmi:
            self.logger.info("Use maximum mutual information, reverse the dialogue")
        self.dataset_path = self.data_path[:-4] + '.pkl'
        if not is_file_exists(self.dataset_path):
            self.logger.info(f"Dataset file doesn't exist, will create a new dataset file save in {self.dataset_path}")
            self.input = self.process_data()
        else:
            self.logger.info(f"Dataset file exists, loading from {self.dataset_path}")
            with open(self.dataset_path, 'rb') as f:
                self.input = pickle.load(f)

    def __getitem__(self, item) :
        input_ids = self.input[item]
        # next-text generation 任务中一个作为输入一个作label
        return input_ids, input_ids

    def __len__(self) :
        return len(self.input)

    def process_data(self) :
        dialogue_len = list()
        dialogue_text = list()
        for index, dialogue in enumerate(tqdm(self.data)) :
            if "\r\n" in dialogue :
                utterances = dialogue.split("\r\n")
            else :
                utterances = dialogue.split("\n")
            input_ids = self.START_TOKEN
            if self.mmi:
                # 使用 maximum mutual information, 需要把每一轮对话的顺序翻转
                utterances = reversed(utterances)
            for utterance in utterances :
                utterance += self.SEP_TOKEN
                input_ids += utterance
            input_ids = input_ids[:self.max_len]
            input_ids = self.tokenizer.encode(
                input_ids,
                add_special_tokens=False,
                return_tensors='pt'
            )
            dialogue_text.append(input_ids)
            dialogue_len.append(input_ids.shape[1])
        dialogue_text = sorted(dialogue_text, key=lambda x : len(x))
        len_mean = np.mean(dialogue_len)
        len_max = np.max(dialogue_len)
        len_min = np.min(dialogue_len)
        self.logger.info("Finish preprocessing data")
        self.logger.info(f"The mean of dialogue length is {len_mean}, the max length of the dialogue is {len_max} "
                         f"and the min length of the dialogue is {len_min}")
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(dialogue_text, f)
        return dialogue_text


def collator_fn(batch) :
    """
    计算一个batch里最长的那个input，然后把其余的sample都pad到这个长度
    """
    input_ids = []
    labels = []
    PAD_TOKEN = [0]
    LABEL_PAD_TOKEN = [-100]
    max_len = 0
    for input_id, _ in batch :
        if max_len < input_id.shape[1] :
            max_len = input_id.shape[1]
    for input_id, label in batch :
        input_len = input_id.shape[1]
        padding = torch.tensor([PAD_TOKEN * (max_len - input_len)])
        label_padding = torch.tensor([LABEL_PAD_TOKEN * (max_len - input_len)])
        input_id = torch.cat((input_id, padding), 1)
        label = torch.cat((label, label_padding),1)
        input_ids.append(input_id)
        labels.append(label)
    return torch.cat(input_ids).long(), torch.cat(labels).long()


if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import dataloader

    ds = MyDataset(BertTokenizer.from_pretrained('data/pretrained_model/gpt2-chinese'),'data/train_small.txt','data/log_test.txt',300, True)
    dl = dataloader.DataLoader(ds, batch_size=3, num_workers=2, shuffle=True, collate_fn=collator_fn)
    for i,j in dl:
        import ipdb;ipdb.set_trace()
        print(i,j)