import pandas as pd
import torch
from transformers import T5Tokenizer, T5Config
from torch.utils.data import Dataset, DataLoader
import spacy
spacy_nlp = spacy.load("en_core_web_sm")

config = {
    "end_token" : "</s>",
    "start_ans" : "<extra_id_1>",
    "end_ans" : "<extra_id_1>",
    "batch_size" : 8,
    "max_len_context" : 512,
    "max_len_question" : 20,
    "task" : "ask_question",
    "epoch" : 2,
    "path_t5_question_generation" : "./pre-trained-model/t5-question-generation/",
    "learning_rate" : 5e-5,
    "schedule_rate" : 0.83,
    "period_decay" : 3,
    "accumulation_step" : 256,
}

config_ask_question = {
    "early_stopping": True,
    "max_length" : 20,
    "min_length" : 3,
    "num_beams" : 4,
    "prefix" : "ask_question"
}

# Tokenizer
class Tokenizer:
    def __init__(self, ):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def encode_context(self, context, start_ans, end_ans):
        """
            this functin is used to encode the context with highlighting the positions of answer
            format of input context : "aks_question : context <start_ans> answer <end_answer> context </s>"

            return dict{input_ids, attention_mask, token_type_ids}
        """
        before_ans = context[:start_ans]
        ans = context[start_ans:end_ans]
        after_ans = context[end_ans:]
        input = config['task'] + before_ans + config['start_ans'] + ans + config['end_ans'] + after_ans + " " + config[
            'end_token']
        output = self.tokenizer.encode_plus(input, max_length=config['max_len_context'], pad_to_max_length=True)
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


# dataset
class QDataset(Dataset):

    def __init__(self, filename, tokenizer):
        self.data = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.shuffle()

    def shuffle(self):
        self.data = self.data.sample(frac=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.data.iloc[idx]
        context, question, start_ans, end_ans = item['context'], item['question'], item['start_ans'], item['end_answer']
        input_dict, output_dict = self.tokenizer.encode_context(context, start_ans,
                                                                end_ans), self.tokenizer.encode_question(question)
        return input_dict['input_ids'], input_dict['attention_mask'], output_dict['input_ids'], output_dict[
            'attention_mask']

configT5_model = T5Config.from_pretrained('t5-base')
configT5_model.task_specific_params[config['task']] = config_ask_question
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up tokenizer and dataset
tokenizer = Tokenizer()
train_dataset = QDataset('./data/squad2.0/train_squad2.0.csv', tokenizer)
valid_dataset = QDataset('./data/squad2.0/valid_squad2.0.csv', tokenizer)

train_loader, valid_loader = DataLoader(train_dataset, batch_size=config['batch_size']), DataLoader(valid_dataset, batch_size=config['batch_size'])