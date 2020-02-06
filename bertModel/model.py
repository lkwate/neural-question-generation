import torch
import json
import random
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize

## import about transformer model
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math 
from torch.nn.init import xavier_uniform_
from torch import optim
from pytorch_pretrained_bert import BertModel


from ..utils import *

#Bert fine tuned modèle.
class BertQG(nn.Module):
    
    def __init__(self, path_to_bert_model, d_model = 768, vocab_size = 30523):
        super(BertQG, self).__init__()
        self.bert = BertModel.from_pretrained(path_to_bert_model).cuda(device)
        self.linear = nn.Linear(d_model, vocab_size).cuda(device)
        self.logsoftmax = nn.LogSoftmax(dim = -1).cuda(device)
        
    def forward(self, indexed_tokens, segments_ids, questions_ids):

        indexed_tokens = indexed_tokens + questions_ids + [constant['mask']]
        segments_ids = segments_ids + [1] * (len(questions_ids) + 1)
        
        #convert to tensor
        indexed_tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_ids_tensor = torch.tensor([segments_ids]).to(device)
        
        #encoder inputs 

        encoders_layers, _ = self.bert(indexed_tokens_tensor, segments_ids_tensor)
        # last output of the last hidden layers
        output = encoders_layers[-1].squeeze(0)[-1, :].unsqueeze(0).to(device) #shape (768)
        output = self.linear(output)
        output = self.logsoftmax(output)
        
        return output
    
    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1: 
                xavier_uniform_(p)