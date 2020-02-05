## import about transformer model
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, TransformerDecoderLayer, TransformerDecoder, Transformer
import math 
from torch.nn.init import xavier_uniform_
from torch import optim

# Transformers
## Transformer model
''''
* sourceEmbeddingLayer
* PositionEncoderLayer
* GeneratorLayer
* Transformer
* GreedyDecoder
''''

## Embedding Layer
class EmbeddingLayer(nn.Module):
    
    def __init__(self, d_model=768, path_to_bert_tokenizer = None, path_to_bert_model = None):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model 
        self.bertEmb = BertEmb(path_to_bert_tokenizer, path_to_bert_model)
        
    def forward(self, sentence):
        
        # form of result : (encoders , padding_length): 
        #     encoders : embedding + padding representation of sentence 
        #     padding_length : length of padding sequence added
        
        '''
            result : encoders_layers, index_padding 
                encoders_layers : words vector representation of sequence plus pad
                index_padding : index at which begin padding
        '''
        output = self.bertEmb.get_word_emb(sentence)
        output = output * math.sqrt(self.d_model)
        return output.to(device)
    

## Position Encoder 
class PositionEncoder(nn.Module): 
    def __init__(self, d_model, dropout=0.1, max_len=2500):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        #shape of x : (x.size(0), d_model)
        return self.dropout(x)



## Transfomer model
class TransformerModel(nn.Module):
    
    def __init__(self, d_model=768, vocab_size =30522, path_to_bert_tokenizer=None, path_to_bert_model=None):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(d_model=768, dropout=0.3, num_encoder_layers=24, num_decoder_layers=24, nhead=32, dim_feedforward=4096).cuda(device)
        self.embedding = EmbeddingLayer(d_model, path_to_bert_tokenizer, path_to_bert_model).cuda(device)
        self.position_encoder = PositionEncoder(d_model).cuda(device)
        self.linear = nn.Linear(d_model, vocab_size).cuda(device)
        self.log_softmax = nn.LogSoftmax(dim=-1).cuda(device)
        
        self.add_softmax = nn.Softmax(dim = 0).cuda(device)
        #initialize parameter of model
        self._reset_parameters()
        
        
    def forward(self, src, response, tgt):
        #process context
        src = self.embedding(src).to(device)
        src = src.unsqueeze(1).to(device)
        #process response
        response = self.embedding(response).to(device)
        response = response.unsqueeze(1).to(device)
        src = torch.cat((response, src), dim = 0).to(device)
        #position encoder 
        src = self.position_encoder(src).to(device)
        tgt = self.embedding(tgt).to(device)
        tgt = tgt.unsqueeze(1).to(device)
        tgt = self.position_encoder(tgt).to(device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device)
        memory = self.transformer.encoder(src).to(device)

        output = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask).to(device)
        output = output.squeeze(1)
        
        #add attention 
        #output = torch.cat((self.add_attention(memory, output), output), dim = 1)
        output = self.linear(output)
        output = self.log_softmax(output).to(device)
        
        ## memory and probabilities distribution
        return memory, output
    
    def decode(self, tgt, memory):
        tgt = self.embedding.bertEmb.get_word_emb_by_indices(tgt).to(device)
        tgt = tgt.unsqueeze(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0])
        output = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask).to(device)
        output = output.squeeze(1)
        #output = torch.cat((self.add_attention(memory, output), output), dim = 1)
        output = self.linear(output)
        
        output = self.log_softmax(output).to(device)
        
        return output

    def add_attention(self, memory, output): 
        memory = memory.squeeze(1) #shape (S, E)
        output = torch.matmul(memory, output.T)
        output = self.add_softmax(output)
        output = torch.matmul(output.T, memory)
        return output.to(device)
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)