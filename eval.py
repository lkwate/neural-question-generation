from utils import *
import torch
import json

def evaluate(model, context, start_answer, end_answer, tokenizer, max_length = constant['max_question_length']):
    question_predicted = []
    with torch.no_grad():
        indexed_tokens_context, segments_ids_context = tokenizer.processContextAnswer(context, start_answer, end_answer)
        indexed_tokens_question = [constant['cls']]
        segments_ids_question  = [1]
        memory, output = model(indexed_tokens_context, segments_ids_context, indexed_tokens_question, segments_ids_question)
        for i in range(max_length):
            target_index = torch.argmax(output[-1, :]).item()
            question_predicted.append(target_index)
            if target_index == 102:
                break
            indexed_tokens_question.append(target_index)
            segments_ids_question.append(1)
            output = model.decode(memory, indexed_tokens_question, segments_ids_question)

        question_predicted = tokenizer.tokenizer.decode(question_predicted)
    return question_predicted


def load_model(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.required_grad = False

    model.eval()
    return model

def test(model, dataset, tokenizer, output_path):
    data = {}
    data['predictions'] = []
    data['bleu_score'] = 0
    with open(output_path, "a") as file: 
        for data in dataset:
            prediction = {}
            context = data[0]
            start_answer = data[2]
            end_answer = data[3]
            truth_question = data[1]
            question = evaluate(model, context, start_answer, end_answer, tokenizer)
            prediction['True'] = truth_question
            prediction['Predicted'] = question
            prediction['bleu_score'] = 0
            data['predictions'].append(prediction)
        json.dump(data, file)

            
if __name__ == '__main__':

    #load model from checkpoint 
    model = load_model('./checkpoint/checkpointModel.pth')

    #tokenizer
    tokenizer = Tokenizer('./pre_trained/bert_base_uncased_tokenizer/')

    #output path
    output_path = './output/output_question.txt'

    #load train dataset
    dataset = Dataset('./data/squad2.0/dev-v2.0.json')

    #test
    test(model, dataset, tokenizer, output_path)
