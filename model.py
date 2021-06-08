import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, ElectraTokenizer, ElectraModel
from collections import namedtuple
import ray.tune as tune
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import json


class Tokenizer:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_generator)
        self.max_len_context = args.max_len_context
        self.max_len_answer = args.max_len_answer
        self.max_len_question = args.max_len_question
        self.task = args.task
        
    def __call__(self, context=None, answer=None, question=None):
        if context and answer:
            if type(context) is not list:
                context = [context]
            if type(answer) is not list:
                answer = [answer]
                
            # add separate token to answers
            context = [self.task + ' ' + c for c in context]
            answer = [self.tokenizer.eos_token + ' ' + str(ans) for ans in answer]
            
            batch_context = self.tokenizer(context, max_length=self.max_len_context, padding='max_length', truncation=True)
            batch_answer = self.tokenizer(answer, max_length=self.max_len_answer, padding='max_length', truncation=True)
            input_ids = torch.LongTensor([q[:-1] + s for (q, s) in zip(batch_context.input_ids, batch_answer.input_ids)])
            attention_mask = torch.FloatTensor([q[:-1] + s for (q, s) in zip(batch_context.attention_mask, batch_answer.attention_mask)])
            
            return input_ids, attention_mask
        
        if question:
            if type(question) is not list:
                question = [question]
            batch_question = self.tokenizer(question, max_length=self.max_len_question, padding='max_length', truncation=True, return_tensors='pt')
            
            return batch_question.input_ids, batch_question.attention_mask

class SquadDataset(Dataset):
    def __init__(self, args, train=True):
        self.tokenizer = Tokenizer(args)
        if train:
            self.dataset = pd.read_csv(args.train_dataset)
        else:
            self.dataset = pd.read_csv(args.val_dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        raw = self.dataset.iloc[idx]
        context, question, answer = raw['context'], raw['question'], raw['text']
        
        return self.tokenizer(context=context, answer=answer), self.tokenizer(question=question)

class SquadDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args 
        self.tokenizer = Tokenizer(args)
    
    def prepare_data(self):
        self.train_dataset = SquadDataset(self.args, train=True)
        self.test_dataset = SquadDataset(self.args, train=False)
        
    def setup(self, stage):
        val_length = int(self.args.val_test_split_factor * len(self.test_dataset))
        self.val_dataset, self.test_dataset = random_split(self.test_dataset, [val_length, len(self.test_dataset) - val_length])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.args.num_workers, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.args.num_workers, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.args.num_workers, batch_size=self.batch_size)

class Evaluator:
    def __init__(self, args, device):
        self.args = args
        self.tokenizer = ElectraTokenizer.from_pretrained(args.model_evaluator)
        self.model = ElectraModel.from_pretrained(args.model_evaluator).eval().to(device)
        self.alpha = args.alpha
        self.device = device
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def scale_reward(self, reward):
        return (reward + 1 - self.alpha) / (2 - self.alpha)
        
    def __call__(self, sen1, sen2):
        with torch.no_grad():
            input_ids = self.tokenizer([sen1.lower(), sen2.lower()], return_tensors='pt', truncation=True, padding=True).to(self.device)
            out = self.model(**input_ids)['last_hidden_state']
            sim = self.cos(out[0, 0, :], out[1, 0, :]).item()
            sen1_tokens = word_tokenize(sen1)
            sen2_tokens = word_tokenize(sen2)
            bleu = sentence_bleu([sen2_tokens], sen1_tokens)
            reward = bleu * self.alpha + (1 - self.alpha) * sim
            
            return self.scale_reward(reward)
    

class LitQGModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = Evaluator(args, self.device_)
        self.generator = T5ForConditionalGeneration.from_pretrained(args.model_generator)
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_generator)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.gamma = args.gamma
        self.lr = args.learning_rate
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler' : optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, gamma=self.args.lr_scheduler_factor),
            'interval' : 'step',
            'frequency' : 1,
            'strict' : True
        }
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # forward
        loss = self.compute_loss(batch, batch_idx)
        
        optimizer = self.optimizers()
        
        # backward
        self.manual_backward(loss)
        
        if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()
        
            self.log('train_loss', loss, prog_bar=True)
            return loss
        
    
    def forward(self, input_ids, attention_mask):
        output = self.generator.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
        return output
    
    def compute_loss(self, batch, batch_idx):
        input_ids, attention_mask = batch[0][0].squeeze(1), batch[0][1].squeeze(1)
        decoder_input_ids, decoder_attention_mask = batch[1][0].squeeze(1), batch[1][1].squeeze(1)
        out = self.generator(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids)
        loss, logits = out.loss, out.logits
        prob = F.softmax(logits, dim=-1)
        batch_size = input_ids.shape[0]
        loss_tune = torch.zeros(batch_size)
        
        for i in range(batch_size):
            prediction = self.tokenizer.decode(prob[i, :, :].argmax(dim=-1).tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ground_truth = self.tokenizer.decode(decoder_input_ids[i, :].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            r = 1 - self.evaluator(prediction, ground_truth)
            loss_tune[i] = r * self.cross_entropy_loss(logits[i], decoder_input_ids[i])
        
        loss_tune = loss_tune.mean()
        loss = loss * self.gamma + (1 - self.gamma) * loss_tune
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('val_loss', loss, on_step=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('ptl/val_loss', avg_loss)
