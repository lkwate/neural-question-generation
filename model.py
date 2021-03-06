import torch
import torch.nn as nn
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langdetect import detect
import sys
import os
import wget
import shutil


MODEL_PATH = os.getcwd() + "/model/"
if not os.path.exists(MODEL_PATH):
	os.mkdir(MODEL_PATH)
CONFIG_URL = "https://lkwate-model.s3.eu-west-3.amazonaws.com/config.json"
CONFIG_SAVE_PATH = "{}config.json".format(MODEL_PATH)
MODEL_BIN_URL = "https://lkwate-model.s3.eu-west-3.amazonaws.com/model.bin"
MODEL_BIN_SAVE_PATH = "{}pytorch_model.bin".format(MODEL_PATH)

# config
config = {
    "end_token" : "</s>",
    "ans" : "<extra_id_1>",
    "max_len_context" : 512,
    "max_len_question" : 30,
    "task" : "ask_question: "
}


class Tokenizer:
    def __init__(self,):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    def encode_context(self, context, answer):
        """
            this functin is used to encode the context with highlighting the positions of answer
            format of input context : "aks_question : context <ans> answer </s>"
            
            return dict{input_ids, attention_mask, token_type_ids}
        """
        input = config['task'] + context + " "+ config['ans'] + answer + config['end_token']
        output = self.tokenizer.encode_plus(input, max_length = config['max_len_context'], truncation=True, padding=True)
        return output
    
    def encode_question(self, question):
        """
            this function is used to encode the question
            format of input question : "question </s>"
            output : list of input_ids
        """
        input = question + " " + config['end_token']
        output = self.tokenizer.encode_plus(input, max_length=config['max_len_question'], pad_to_max_length=True)
        return output

# model
class QGModel(nn.Module):
    def __init__(self, configT5, path=None):
        super(QGModel, self).__init__()
        if path is not None:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(path, config=configT5)
        else:
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base', config=configT5)
        
    def forward(self, input_ids_ctx, attention_mask_ctx, input_ids_qt = None, attention_mask_qt = None):
        output = self.t5_model(input_ids=input_ids_ctx, attention_mask=attention_mask_ctx, 
                               decoder_attention_mask=attention_mask_qt, lm_labels=input_ids_qt)
        return output
    
    def predict(self, intput_ids_ctx):
        output = self.t5_model.generate(intput_ids_ctx, top_p=0.95, do_sample=True, top_k=30)
        return output



class Model():
    
    def __init__(self):
        
        if not os.path.isfile(CONFIG_SAVE_PATH):
            print("Download config from aws s3... {}".format(CONFIG_URL))
            wget.download(CONFIG_URL, bar= self._bar_custom)
            shutil.move("config.json", "model/config.json")
    
        if not os.path.isfile(MODEL_BIN_SAVE_PATH):
            print("Download model from aws s3... {}".format(MODEL_BIN_URL))
            wget.download(MODEL_BIN_URL, bar=self._bar_custom)
            shutil.move("model.bin", "model/pytorch_model.bin")
            
        #tokenizer
        self.tokenizer = Tokenizer()
        #device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model
        self.model = QGModel(MODEL_PATH, path = MODEL_PATH).to(self.device)     
    
    @staticmethod
    def _bar_custom(current, total, width = 80):
        sys.stdout.write("\r" + "Downloading: %d%% [%d / %d] bytes\n" % (current / total * 100, current, total))
        sys.stdout.flush()
        
    
    def generate(self, context, answer):
        lang = detect(context)
        if lang != "en":
            raise ValueError("Text should be in english")
        
        input_ids = [self.tokenizer.encode_context(context, answer)['input_ids']]
        input_ids = torch.LongTensor(input_ids).to(self.device)
        
        predicted_question = self.model.predict(input_ids).squeeze(0)
        predicted_question = self.tokenizer.tokenizer.decode(predicted_question.tolist())
        
        return predicted_question
