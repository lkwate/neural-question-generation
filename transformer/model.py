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
        
    def forward(self, sentence, start_answer = None, end_answer = None):
        
        # form of result : (encoders , padding_length): 
        #     encoders : embedding + padding representation of sentence 
        #     padding_length : length of padding sequence added
        
        '''
            result : encoders_layers, index_padding 
                encoders_layers : words vector representation of sequence plus pad
                index_padding : index at which begin padding
        '''
        output = self.bertEmb.get_word_emb(sentence, start_answer, end_answer)
        output = output * math.sqrt(self.d_model)
        return output
    

## Position Encoder 
class PositionEncoder(nn.Module): 
    def __init__(self, d_model, dropout=0.1, max_len=2500):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        self.transformer = Transformer(d_model=768, num_encoder_layers=24, num_decoder_layers=24)
        self.embedding = EmbeddingLayer(d_model, path_to_bert_tokenizer, path_to_bert_model)
        self.position_encoder = PositionEncoder(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        #initialize parameter of model
        self._reset_parameters()
        
        
    def forward(self, src, response, tgt, start_answer=None, end_answer=None):
        #process context
        src = self.embedding(src, start_answer, end_answer)
        src = src.unsqueeze(1)
        #process response
        response = self.embedding(response)
        response = response.unsqueeze(1)
        src = torch.cat((src, response), dim = 0)
        #position encoder 
        src = self.position_encoder(src)
        tgt = self.embedding(tgt)
        tgt = tgt.unsqueeze(1)
        tgt = self.position_encoder(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0])
        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.log_softmax(output)
        
        ## memory and probabilities distribution
        return memory, output
    
    def decode(self, tgt, memory):
        tgt = self.embedding.bertEmb.get_word_emb_by_indices(tgt)
        tgt = tgt.unsqueeze(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0])
        output = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask)
        output = output.squeeze(1)
        output = self.linear(output)
        
        output = self.log_softmax(output)
        
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)