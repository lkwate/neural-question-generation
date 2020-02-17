import json
import random
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
#from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize # to tokenize paragraph in sentence
import nltk
nltk.download('punkt')
import torch


#retreive device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#definition of constant
constant = {
    'd_model' : 512,  ## dimension of simple model's features
    'nhead' : 8,   ## number of head in multihead attention
    'max_question_length' : 15,  ## max length of question,
    'number_layer' : 6,  ## depth of stack of layers
    'learning_rate' : 5e-5,  ## learning rate of optimizer
    'vocab_size' : 30522,  ## vocabulary size
    'dropout' :  0.1,  ## dropout hyperparameter for regularization
    'd_emb' : 768, ## dimension of word embeddings provide by bertModel-base-uncased
    'start_answer_token' : 1,  ## token follow by the answer spanned in the context
    'end_answer_token' : 2,  ## the end token of the answer spanned in the context
    'pad' : 0,  ## pad token
    'cls' : 101,  ## cls token, begin token of sequence
    'sep' : 102,  ## separate token
    'mask' : 103,  ## mask token
    'batch_size' : 16, ## batch size
    'epoch' : 2 ## number of times training will be repeat
}

#tokenizer 
class Tokenizer():

    def __init__(self, path_to_tokenizer):
	    ##self.tokenizer = tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(path_to_tokenizer)

    def tokenize(self, input):
        return self.tokenizer.tokenize(input)

    def processContextAnswer(self, context, start_answer, end_answer):
        before_answer = self.tokenizer.encode(context[:start_answer], add_special_tokens=False)
        answer = self.tokenizer.encode(context[start_answer: end_answer], add_special_tokens=False)
        after_answer = self.tokenizer.encode(context[end_answer:], add_special_tokens=False)
        indexed_tokens = [constant['cls']] + before_answer + [constant['start_answer_token']] + answer + [
            constant['end_answer_token']] + after_answer + [constant['sep']]
        segments_tokens = [1] * len(indexed_tokens)

        return indexed_tokens, segments_tokens

    def processQuestion(self, question):
        question_tokens = [constant['cls']] + self.tokenizer.encode(question, add_special_tokens=False) + [
            constant['sep']]
        question_segments = [1] * len(question_tokens)

        return question_tokens, question_segments

    def decode(self, input_ids):
        output = self.tokenizer.convert_ids_to_tokens(input_ids)
        output = self.tokenizer.convert_tokens_to_string(output)
        return output

#building of dataset based on SUAD2.0
# dataset
# dataset
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
                                if not qas['is_impossible']:
                                    question = qas['question']

                                    # ignore question which length is less than 3
                                    if len(question) <= 2:
                                        continue

                                    # loop over answers
                                    length_answer = -1
                                    start_answer = 10e4
                                    end_answer = None
                                    for ans in qas['answers']:
                                        if ans['answer_start'] < start_answer:
                                            start_answer = ans['answer_start']
                                        if length_answer <= len(ans['text']):
                                            end_answer = start_answer + len(ans['text'])
                                        response = context[start_answer: end_answer]

                                    if response == '':
                                        continue

                                    index = 0
                                    for sentence in sent_tokenize(context):
                                        j = sentence.find(response)
                                        if j != -1 and index <= start_answer and (index + len(sentence)) >= end_answer:
                                            self.dataset.append((sentence, question, j, j + len(response)))
                                            break
                                        index += len(sentence)
        # shuffle item in dataset
        random.shuffle(self.dataset)
