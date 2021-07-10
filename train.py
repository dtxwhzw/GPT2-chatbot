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


class Trainer(object) :
    def __init__(self, args) :
        self.conf = args
        self.logger = create_logger(get_real_path(self.conf.log_path))
        self.gpt_path = getattr(self.conf, 'gpt_path', None)
        self.gpt_path = get_real_path(self.gpt_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.gpt_path)
        self.model = Gpt2Generator(self.conf)
        self.device = self.conf.device
        if self.device != 'cuda' :
            self.logger.info("[Warning] Use cpu training!")
        self.cuda_ids = self.conf.device_ids
        self.logger.info(f'Using device: {self.device}')
        self.model.to(self.device)
        self.lr = self.conf.lr
        self.mmi = self.conf.use_mmi
        self.max_len = self.conf.max_len
        self.eps = self.conf.eps
        self.checkpoint = get_real_path(self.conf.checkpoint_path)
        self.checkpoint_dir = os.path.dirname(self.checkpoint)
        self.resume = self.conf.resume
        self.writer = SummaryWriter(log_dir=get_real_path(self.conf.writer_path))

        # 并行训练cd
        if self.device == 'cuda' and len(self.cuda_ids) > 1 :
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.logger.info(f"use GPU {self.cuda_ids} to train.")

        num_parameters = 0
        parameters = self.model.parameters()
        for parameter in parameters :
            num_parameters += parameter.numel()
        self.logger.info('The number of the model parameters is {}'.format(num_parameters))

        # 记录参数设置
        self.logger.info("Configuration is {}".format(self.conf))

    def train(self, data_path, epochs, model_path, batch_size) :
        train_loader, val_loader = self.get_data_loader(data_path, batch_size)
        optimizer = AdamW(self.model.parameters(),
                          lr=self.lr,
                          eps=self.eps
                          )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.conf.warm_up_steps,
                                                    num_training_steps=total_steps)
        if not self.resume :
            self.logger.info("Start Training")
            train_losses, val_losses = [], []
            best_val_loss = float('inf')
            best_val_acc = -float('inf')
            start_epoch = 0
        else :
            savepoint = torch.load(self.checkpoint)
            self.model.load_state_dict(savepoint['model'])
            optimizer.load_state_dict(savepoint['optimizer'])
            scheduler.load_state_dict(savepoint['scheduler'])
            train_losses, val_losses = savepoint['train_loss'], savepoint['val_loss']
            best_val_loss = savepoint['best_loss']
            best_val_acc = savepoint['best_acc']
            start_epoch = savepoint['epoch'] + 1
            self.logger.info(f"Resume training from epoch {start_epoch}")

        # 开始训练
        for epoch in tqdm(range(start_epoch, epochs)) :
            # ============== train ================ #
            train_loss, train_acc = self.train_epoch(train_dataloader=train_loader,
                                                     optimizer=optimizer,
                                                     scheduler=scheduler,
                                                     epoch=epoch)
            train_losses.append(train_loss)

            # ============== validate ================ #
            validate_loss, validate_acc = self.evaluate(val_loader, epoch)
            val_losses.append(validate_loss)

            if validate_loss < best_val_loss :
                best_val_loss = validate_loss
                self.logger.info(f"Saving current best model with minimum loss for epoch {epoch + 1}")
                best_model_path = os.path.join(model_path, 'mini_loss_model.cpt')
                torch.save(self.model.state_dict(), best_model_path)
            if validate_acc > best_val_acc :
                best_val_acc = validate_acc
                self.logger.info(f"Saving current best model with best accuracy for epoch {epoch + 1}")
                best_model_path = os.path.join(model_path, 'best_acc_model.cpt')
                torch.save(self.model.state_dict(), best_model_path)

            # 保存最后一个epoch的进度
            checkpoint = {
                'model' : self.model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'epoch' : epoch,
                'train_loss' : train_losses,
                'val_loss' : val_losses,
                'best_loss' : best_val_loss,
                'best_acc' : best_val_acc
            }
            if not self.checkpoint_dir :
                mk_folder_for_file(self.checkpoint_dir)
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
            former_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch - 1}.pt")
            if os.path.exists(former_checkpoint_path) :
                self.logger.info(f"delect checkpoint {former_checkpoint_path}")
                os.popen(f"rm {former_checkpoint_path}")
            self.logger.info(f"save checkpoint in {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)

        self.logger.info("Finish Training.")
        self.logger.info(f"train_losses: {train_losses}")
        self.logger.info(f"validate_losses: {val_losses}")

    def train_epoch(self, train_dataloader, optimizer, scheduler, epoch) :
        epoch_start_time = datetime.now()
        self.model.train()
        # 记录下整个epoch的loss
        total_loss = 0.0
        # epoch_correct_num: 每个epoch中, output预测正确的word的数量
        # epoch_total_num: 每个epoch中, output预测的word的数量
        epoch_correct_num, epoch_total_num = 0, 0

        for batch_idx, (input_ids, labels) in enumerate(tqdm(train_dataloader)) :
            try :
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss, logits = self.model(input_ids, labels=labels)
                loss = loss.mean()
                batch_correct, batch_total_num = calculate_acc(logits, labels, )
                epoch_correct_num += batch_correct
                epoch_total_num += batch_total_num

                batch_acc = batch_correct / batch_total_num

                total_loss += loss.item()

                if self.conf.gradient_accumulation_steps :
                    loss = loss / self.conf.gradient_accumulation_steps

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.max_grad_norm)

                # 进行一定step的梯度累积之后，更新参数
                if (batch_idx + 1) % self.conf.gradient_accumulation_steps == 0 :
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (batch_idx + 1) % self.conf.log_step == 0 :
                    self.logger.info(
                        "batch {} of epoch {}, loss {}, batch accuracy {}, learning rate {}".format(
                            batch_idx + 1, epoch + 1, loss.item() * self.conf.gradient_accumulation_steps, \
                            batch_acc, scheduler.get_lr()
                        )
                    )
                    self.writer.add_scalar('loss', loss.item(), epoch + 1)
            except RuntimeError as exception :
                if "out of memory" in str(exception) :
                    self.logger.info("WARNING! RUN OUT OF MEMORY")
                    if hasattr(self, 'device', 'empty_cache') :
                        self.device.empty_cache()
                else :
                    self.logger.info(str(exception))
                    raise exception

        epoch_mean_loss = total_loss / len(train_dataloader)
        epoch_mean_acc = epoch_correct_num / epoch_total_num
        self.logger.info("Train epoch {}: loss {}, accuracy :{}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

        self.logger.info("epoch {} finished".format(epoch + 1))
        epoch_finish_time = datetime.now()
        self.logger.info("Time for train epoch {}: {}".format(epoch + 1, epoch_finish_time - epoch_start_time))

        return epoch_mean_loss, epoch_mean_acc

    def evaluate(self, val_loader, epoch) :
        self.logger.info("Start evaluate.")
        self.model.eval()
        start_time = datetime.now()
        total_loss = 0.0
        epoch_correct_num, epoch_total_num = 0, 0
        with torch.no_grad() :
            for batch_idx, (input_ids, labels) in enumerate(tqdm(val_loader)) :
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss, logits = self.model(input_ids, labels)
                loss = loss.mean()
                batch_correct_num, batch_total_num = calculate_acc(logits, labels)
                epoch_correct_num += batch_correct_num
                epoch_total_num += batch_total_num
                # 计算该batch的accuracy

                total_loss += loss.item()

            epoch_mean_loss = total_loss / len(val_loader)
            epoch_mean_acc = epoch_correct_num / epoch_total_num
            finish_time = datetime.now()
            self.logger.info(
                "Evaluate epoch {}: loss {}, accuracy {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))
            self.logger.info("Time for evaluate epoch {}: {}".format(epoch + 1, finish_time - start_time))
            return epoch_mean_loss, epoch_mean_acc

    def get_data_loader(self, file_path, batch_size) :
        np.random.seed(42)
        ds = MyDataset(self.tokenizer, file_path, self.logger, self.max_len, self.mmi)
        length = len(ds)
        length_idx = [int(length * 0.7), length - int(length * 0.7)]
        train_ds, val_ds = random_split(
            ds,
            lengths=length_idx
        )
        train_loader = dataloader.DataLoader(train_ds, batch_size=batch_size, num_workers=self.conf.nuw_workers,
                                             shuffle=True, collate_fn=collator_fn)
        val_loader = dataloader.DataLoader(val_ds, batch_size=batch_size, num_workers=self.conf.nuw_workers,
                                           shuffle=True, collate_fn=collator_fn)
        return train_loader, val_loader


def main(args) :
    trainer = Trainer(args)
    data_path = get_real_path(args.data_path)
    epochs = args.epochs
    model_path = get_real_path(args.model_path)
    if not os.path.exists(model_path) :
        os.mkdir(model_path)
    batch_size = args.batch_size
    trainer.train(data_path, epochs, model_path, batch_size)


if __name__ == '__main__' :
    import sys, json

    f_conf = parse_args(sys.argv[1])
    main(f_conf)
