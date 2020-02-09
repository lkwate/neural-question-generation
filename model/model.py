## import about transformer model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_
from torch import optim
from pytorch_pretrained_bert import BertModel
from torch.nn import Transformer
from torch.nn.init import xavier_uniform_
from ..utils import *



#position encoding
class PositionEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2500):
        super(PositionEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout).cuda(device)
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term = div_term.to(device)
        pe[:, 0::2] = torch.sin(position * div_term).to(device)
        pe[:, 1::2] = torch.cos(position * div_term).to(device)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # shape of x : (x.size(0), d_model)
        return self.dropout(x).to(device)


#embedding layer
class Embedding(nn.Module):

    def __init__(self, path_to_model, d_model=512):
        super(Embedding, self).__init__()

        self.d_model = d_model
        # layer to embed tokens with BERT
        self.bertModel = BertModel.from_pretrained(path_to_model).cuda(device)
        # layer to projectize output of BERT on space of input of Transformer
        self.linear = nn.Linear(constant['d_emb'], d_model).cuda(device)

    def forward(self, indexed_tokens, segments_ids):
        # get words embedding
        encoders_layers, _ = self.bertModel(indexed_tokens, segments_ids)
        # get the output of the last hidden layers of model
        encoders_layers = encoders_layers[-1].squeeze(0).to(device)
        # projectize the result on dimension of model
        encoders_layers = self.linear(encoders_layers)
        # mutiply by the square root of the d_model dimension of model
        encoders_layers = encoders_layers * torch.sqrt(torch.FloatTensor([constant['d_model']]).to(device))
        # transpose (0, 1) to change the shape to (length_sequence, batch, 512)
        return encoders_layers

#TransformerModel
class TransformerModel(nn.Module):

    def __init__(self, path_to_model, d_model=512, vocab_size=30522):
        super(TransformerModel, self).__init__()

        self.transformer = Transformer(d_model=d_model, nhead=constant['nhead'], dropout=constant['dropout'],
                                       num_encoder_layers=constant['number_layer'],
                                       num_decoder_layers=constant['number_layer']).cuda(device)
        # position encoder
        self.position_encoder = PositionEncoder(d_model).cuda(device)
        # embedding layer for vector representation of the combination of context and answer
        self.embedding_context_answer = Embedding(path_to_model, d_model).cuda(device)
        # embedding layer for vector representation of the question ongoing generated
        self.embedding_question = Embedding(path_to_model, d_model).cuda(device)
        # layer for the projection of output of transformer decoder on the space which the dimension is the size of vocabulary
        self.linear = nn.Linear(d_model, vocab_size).cuda(device)
        # compute log of the probality distribution
        self.log_softmax = nn.LogSoftmax(dim=-1).cuda(device)

        #init parameter
        self._reset_parameter()

    def forward(self, indexed_tokens_context, segments_ids_context, indexed_tokens_question, segments_ids_question):
        # convert to indices and segment to tensor
        indexed_tokens_context = torch.tensor([indexed_tokens_context]).to(device)
        segments_ids_context = torch.tensor([segments_ids_context]).to(device)
        indexed_tokens_question = torch.tensor([indexed_tokens_question]).to(device)
        segments_ids_question = torch.tensor([segments_ids_question]).to(device)

        # get embedding representation of context and answer combination
        src = self.embedding_context_answer(indexed_tokens_context, segments_ids_context)

        src = src.unsqueeze(1)
        # add position features
        src = self.position_encoder(src)

        # get embedding representation of question
        tgt = self.embedding_question(indexed_tokens_question, segments_ids_question)

        tgt = tgt.unsqueeze(1)
        # add position features
        tgt = self.position_encoder(tgt)
        # generate mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device)
        # compute memory
        memory = self.transformer.encoder(src)
        # output
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.log_softmax(output)

        # return memory, and log of probabilities distribution of next tokens
        return memory, output

    def decode(self, memory, indexed_tokens, segments_ids):
        indexed_tokens = torch.tensor([indexed_tokens]).to(device)
        segments_ids = torch.tensor([segments_ids]).to(device)
        tgt = self.embedding_question(indexed_tokens, segments_ids)
        tgt = tgt.unsqueeze(1)
        tgt = self.position_encoder(tgt)
        # generate mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.log_softmax(output)

        return output
    
    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1: 
                xavier_uniform_(p)