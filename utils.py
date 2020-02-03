import torch
import spacy
import en_core_web_sm
import json
import re 
import unicodedata
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


# definition of constant 
constant = {
    'd_model_complex' : 800,     ## dimension of complex model's features
    'd_model_simple' : 768,  ## dimension of simple model's features
    'd_inp_complex' : 800,  ## dimension of linear projection of query, key, and value vector : complex model
    'd_inp_simple' : 768,  ## dimension of linear projection of query, key, and value vector : simple model
    'nhead' : 8,   ## number of head in multihead attention
    'nlayer' : 6,  ## number of layer in encoder and decoder
    'max_question_length' : 40,  ## max length of question, 
    'learning_rate' : 0.01,  ## learning rate of optimizer
    'vocab_size' : 30522,  ## vocabulary size
    'ner_size' : 37, ## size of vocabulary of ner tag
    'pos_size' : 107, ## size of vocabulary of pos tag
    'd_pos' : 16,  ## dimension of embedding representation of pos tagging
    'd_ner' : 16,  ## dimension of embedding representation of ner tagging
    'd_emb' : 768, ## dimension of word embeddings provide by bertModel-base-uncased
    'start_answer_token' : "[unused0]",  ## token follow by the answer spanned in the context 
    'end_answer_token' : "[unused1]",  ## the end token of the answer spanned in the context
    'pad' : 0,  ## pad token
    'cls' : 101,  ## cls token, begin token of sequence
    'sep' : 102,  ## separate token
}

# bert model for word embedding
class BertEmb():
    def __init__(self, path_to_tokenizer = None, path_to_model = None):
        
        if path_to_tokenizer:
            try :
                #self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', path_to_tokenizer)
                self.tokenizer = BertTokenizer.from_pretrained(path_to_tokenizer)
            except : 
                print("Error of loading tokenizer from local file")
        else :
            #self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if path_to_model: 
            try:
                #self.bertModel = torch.hub.load('huggingface/pytorch-transformers', 'model', path_to_model)
                self.bertModel = BertModel.from_pretrained(path_to_model)
            except: 
                print("Error of loading Bertmodel from local file")
        else :
            #self.bertModel = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            self.bertModel = BertModel.from_pretrained('bert-base-uncased')
            
    def get_id_tokens(self, sentence):
        """
            encode a given sentence by given its corresponding ids in dictionary
        """
        #return self.tokenizer.encode(sentence, add_special_tokens = True)
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens
    
    def encode(self, sentence):
        tokenized_text = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens
    
    def tokenize(self, sentence): 
        """
            tokenization of a given sentence
        """
        return self.tokenizer.tokenize(sentence)
    
    def insert_answers_indices(self, sentence, start_answer, end_answer): 
        """
            insert the answers boundaries id token in a sequence
        """
        before_answer = self.encode(sentence[:start_answer])
        answer = self.encode(sentence[start_answer:end_answer])
        after_answer = self.encode(sentence[end_answer:])
        result = [101]
        result += before_answer + [1] + answer + [2] + after_answer + [102]
        return result
    
    
    '''
    def padding_sequence(self, previous_indexed_tokens): 
        """
            assumptions : within this function we consider len(previous_indexed_tokens) < max_input_length
            padding a given sequence by adding the index of token '[PAD]' in dictionary 
            outputs : indexed_tokens, len(previous_indexed_tokens)
        """
        previous_length = len(previous_indexed_tokens)
        padding_length = self.max_input_length - previous_length
        
        # padding
        indexed_tokens = previous_indexed_tokens + [0] * padding_length
        
        return indexed_tokens, previous_length
    '''
    
    def get_word_emb(self, sentence, start_answer = None, end_answer = None):
        """
            retreiving of word embeddings of the tokens of a given sentence by processing
            sentence through BERTModel
        """
        if start_answer and end_answer: 
            indexed_tokens = self.insert_answers_indices(sentence, start_answer, end_answer)
        else :
            indexed_tokens = self.get_id_tokens(sentence)
        """
        #padding sequence if it's necessary
        if len(indexed_tokens) < self.max_input_length:
            indexed_tokens, index_padding = self.padding_sequence(indexed_tokens)
        """
        
        
        return self.get_word_emb_by_indices(indexed_tokens)

    def get_word_emb_by_indices(self, indexed_tokens): 
        
        
        segments_tensor = torch.tensor([[1] * len(indexed_tokens)])
        indexed_tokens_tensor = torch.tensor([indexed_tokens])

        #due to the fact that bertModel cannot take over 512 token, it's necessary to breakdown
        #segments_tensor and indexed_tokens into minibatch of size 500
        length = len(indexed_tokens)  # length of input
        batch_size = 500  # batch size 
        nbatch = length // batch_size  # number of minibatch 
        remain = length != batch_size * nbatch
        
        output = []
        for i in range(nbatch): 
            batch_indexed_tokens_tensor = indexed_tokens_tensor[:, i * batch_size : (i + 1) * batch_size]
            batch_segments_tensor = segments_tensor[:, i * batch_size : (i + 1) * batch_size]
            # process segments_tensor and indexed_tokens through bertModel
            with torch.no_grad(): 
                encoders_layers, _ = self.bertModel(batch_indexed_tokens_tensor, batch_segments_tensor)
                encoders_layers = encoders_layers[-1].squeeze(0)
                
            output.append(encoders_layers)
        
        if remain :
            batch_indexed_tokens_tensor = indexed_tokens_tensor[:, nbatch * batch_size : length]
            batch_segments_tensor = segments_tensor[:, nbatch * batch_size : length]
            with torch.no_grad(): 
                encoders_layers, _ = self.bertModel(batch_indexed_tokens_tensor, batch_segments_tensor)
                encoders_layers = encoders_layers[-1].squeeze(0)
            output.append(encoders_layers)
            
        #concatenate (along the dimension 0 ) the list of output tensor 
        encoders_layers = torch.cat(output, dim = 0)
        ## the result have the shape : (length_tokens, 768)
        encoders_layers.unsqueeze(1)
        return encoders_layers


##build vocabulary 
class Dictionary(): 
    
    def __init__(self, path_to_tag_json, path_to_word_vocab): 
        self.ner_ids = {}
        self.ids_ner = {}
        self.pos_ids = {}
        self.ids_pos = {}
        self.word_ids = {}
        self.ids_word = {}
        
        ## process data of tag (Ner + Pos)
        with open(path_to_tag_json) as json_file: 
            data = json.load(json_file)
            pos_data = data['pos']
            ner_data = data['ner']
            
            self.pos_ids['O'] = 0
            self.ids_pos[0] = 'O'
            index = 1
            #process data of tag
            for pos in pos_data['tags']: 
                self.pos_ids['B-' + pos] = index
                self.ids_pos[index] = 'B-' + pos
                index += 1 
                self.pos_ids['I-' + pos] = index 
                self.ids_pos[index] = 'I-' + pos
                index += 1
            
            self.ner_ids['O'] = 0
            self.ids_ner[0] = 'O'
            index = 1
            #process data of ner 
            for ner in ner_data['tags']: 
                self.ner_ids['B-' + ner] = index 
                self.ids_ner[index] = 'B-' + ner
                index += 1 
                self.ner_ids['I-' + ner] = index
                self.ids_ner[index] = 'I-' + ner
                index += 1
        
        ## building of word dictionary
        index = 0
        with open(path_to_word_vocab) as file: 
            for line in file: 
                line = line.strip()
                self.word_ids[line] = index 
                self.ids_word[index] = line 
                index += 1
                
    def word_by_id(self, id): 
        print(self.ids_word[id])
        return self.ids_word[id]
        
    def id_by_word(self, word): 
        return self.word_ids[word]
        
    def id_by_pos(self, pos): 
        return self.pos_ids[pos]
        
    def pos_by_id(self, id):
        return self.ids_pos[id]
        
    def id_by_ner(self, ner): 
        return self.ner_ids[ner]
        
    def ner_by_id(self, id): 
        return self.ids_ner[id]
    
    def size_ner(self):
        return len(self.ner_ids)
    
    def size_pos(self):
        return len(self.pos_ids)
    
    def size_vocab(self):
        return len(self.word_ids)


## Building of dataset based on SQUAD 2.0
class Dataset(): 
    '''
        rule of validation of data in dataset: 
            context cannot be null
            answer a question regarding the context cannot be impossible
    '''
    def __init__(self, path_to_dataset): 
        self.dataset = {}
        index = -1
        with open(path_to_dataset) as json_file: 
            
            data = json.load(json_file)
            for batch_data in data['data']: 
                index += 1
                self.dataset[index] = {}
                for paragraph in batch_data['paragraphs']:
                    if paragraph['context']: 
                        self.dataset[index]['context'] = paragraph['context']
                        # loop over question and answers for a given context
                        if len(paragraph['qas']) != 0:
                            self.dataset[index]['qas'] = []
                            for qas in paragraph['qas']: 
                                if qas['is_impossible'] is False: 
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
                                    self.dataset[index]['qas'].append((question, start_answer, end_answer))

