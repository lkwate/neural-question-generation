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


#retreive device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#definition of constant
constant = {
    'd_model_complex' : 800,     ## dimension of complex model's features
    'd_model_simple' : 768,  ## dimension of simple model's features
    'd_inp_complex' : 800,  ## dimension of linear projection of query, key, and value vector : complex model
    'd_inp_simple' : 768,  ## dimension of linear projection of query, key, and value vector : simple model
    'nhead' : 8,   ## number of head in multihead attention
    'nlayer' : 6,  ## number of layer in encoder and decoder
    'max_question_length' : 16,  ## max length of question, 
    'learning_rate' : 5e-5,  ## learning rate of optimizer
    'vocab_size' : 30523,  ## vocabulary size
    'ner_size' : 37, ## size of vocabulary of ner tag
    'pos_size' : 107, ## size of vocabulary of pos tag
    'd_pos' : 16,  ## dimension of embedding representation of pos tagging
    'd_ner' : 16,  ## dimension of embedding representation of ner tagging
    'd_emb' : 768, ## dimension of word embeddings provide by bertModel-base-uncased
    'start_answer_token' : 1,  ## token follow by the answer spanned in the context 
    'end_answer_token' : 2,  ## the end token of the answer spanned in the context
    'pad' : 0,  ## pad token
    'cls' : 101,  ## cls token, begin token of sequence
    'sep' : 102,  ## separate token
    'mask' : 103,  ## mask token
    'unused0' : 1
}

#tokenizer 
class Tokenizer():
    
    def __init__(self, path_to_tokenizer):
        self.tokenizer = BertTokenizer.from_pretrained(path_to_tokenizer)
        self.tokenizer.add_tokens(['[HL]'])
    
    def tokenize(self, input):
        return self.tokenizer.tokenize(input)
    
    def convert_tokens_to_ids(self, input):
        return self.tokenizer.convert_tokens_to_ids(self.tokenize(input))
    
    def processContextAnswer(self, context, start_answer, end_answer): 
        before_answer = self.convert_tokens_to_ids(context[:start_answer])
        answer = self.convert_tokens_to_ids(context[start_answer : end_answer])
        after_answer = self.convert_tokens_to_ids(context[end_answer:])
        indexed_tokens = [constant['cls']] + before_answer + [constant['unused0']] + answer + [constant['unused0']] + after_answer + [constant['sep']]
        segments_tokens = [0] * len(indexed_tokens)
        return indexed_tokens, segments_tokens
    
    def processQuestion(self, question):
        question = self.convert_tokens_to_ids(question) + [constant['sep']]
        return question


#building of dataset based on SUAD2.0
class Dataset(): 
    '''
        rule of validation of data in dataset: 
            context cannot be null
            answer a question regarding the context cannot be impossible
    '''
    def __init__(self, path_to_dataset): 
        self.dataset = []
        with open(path_to_dataset) as json_file: 

            data = json.load(json_file)
            for batch_data in data['data']: 
            
                for paragraph in batch_data['paragraphs']:
                    if paragraph['context']: 
                        context = paragraph['context']
                        
                        # loop over question and answers for a given context
                        if len(paragraph['qas']) != 0:
                            for qas in paragraph['qas']: 
                                if not qas['is_impossible'] : 
                                    question = qas['question']
                                    # loop over answers
                                    length_answer = -1
                                    start_answer = 10e4
                                    end_answer = None
                                    for ans in qas['answers']: 
                                        if ans['answer_start'] < start_answer:
                                            start_answer = ans['answer_start']
                                        if length_answer <= len(ans['text']): 
                                            end_answer = start_answer + len(ans['text'])
                                        response = context[start_answer : end_answer]
                                        if response == '':
                                            continue
                                        index = 0
                                        for sentence in sent_tokenize(context):
                                            j = sentence.find(response)
                                            if j != -1 and index <= start_answer and (index + len(sentence))>= end_answer: 
                                                self.dataset.append((sentence, question, j, j + len(response)))
                                                break
                                            index += len(sentence)


#Build vocabulary 
class Dictionary(): 
    
    def __init__(self, path_to_word_vocab): 
        
        self.word_ids = {}
        self.ids_word = {}
        
        ## building of word dictionary
        index = 0
        with open(path_to_word_vocab) as file: 
            for line in file: 
                line = line.strip()
                self.word_ids[line] = index 
                self.ids_word[index] = line 
                index += 1
                
    def word_by_id(self, id): 
        return self.ids_word[id]
        
    def id_by_word(self, word): 
        return self.word_ids[word]