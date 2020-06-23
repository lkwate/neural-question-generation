from utils import valid_loader, tokenizer, device, configT5_model, config, spacy_nlp
from model import QGModel
import torch
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
import json
from langdetect import detect


def bleu(candidate, reference):
    candidate = [candidate.lower().split()]
    reference = [[reference.lower().split()]]
    return bleu_score(candidate, reference)


def evaluate(model, data_loader, tokenizer, device):
    predictions = []
    score = 0
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader)):
            input_ids_ctx = torch.stack(batch[0], dim=1).to(device)
            input_ids_qt = torch.stack(batch[2], dim=1).to(device)
            output = model.predict(input_ids_ctx)

            for k in range(output.shape[0]):
                ground_truth = tokenizer.tokenizer.decode(input_ids_qt[k].tolist())
                predicted_question = tokenizer.tokenizer.decode(output[k].tolist())
                temp_score = bleu(predicted_question, ground_truth)
                score += temp_score
                predictions.append((ground_truth, predicted_question, temp_score))
        score /= len(predictions)
        predictions = {
            "score" : score,
            "predictions" : predictions
        }

    return predictions


def generate(model, tokenizer, context, device, list_dict_answers=None):
    # check the length of context
    if len(tokenizer.tokenizer.tokenize(context)) >= config['max_len_context'] - 4:
    	# send {'length_failed' : True}
        return {'invalid_length' : True}

    # check whether the context is in english
    lang = detect(context)
    if lang != 'en':
    	# send {'language_failed' : True}
        return {'invalid_lang' : True}

    # define inputs of the model
    input_ids = []

    # preprocessing
    # name entity extraction
    ner = spacy_nlp(context)
    batch_size = len(ner.ents)
    answers = []
    for ent in ner.ents:
        temp_dict = tokenizer.encode_context(context, ent.start_char, ent.end_char)
        input_ids.append(temp_dict['input_ids'])
        answers.append(ent.text)

    if list_dict_answers is not None:
        for item in list_dict_answers:
            start_ans, end_ans = item['start_ans'], item['end_ans']
            answer = context[start_ans:end_ans]
            tem_dict = tokenizer.encode_context(context, start_ans, end_ans)
            input_ids.append(temp_dict['input_ids'])
            answers.append(answer)
            batch_size += 1

    input_ids = torch.LongTensor(input_ids).to(device)

    # predict question
    results = []
    for k in range(batch_size):
        predicted_question = model.predict(input_ids[k].unsqueeze(0).to(device)).squeeze(0)
        results.append(
            {
                "question": tokenizer.tokenizer.decode(predicted_question.tolist()),
                "answer": answers[k]
            }
        )

    return results

if __name__ == '__main__':
    model = QGModel(configT5_model, config['path_t5_question_generation']).to(device)
    predictions = evaluate(model, valid_loader, tokenizer, device)
    with open('./prediction/prediction.json', 'w') as file:
        json.dumps(predictions, file)
