from utils import *
import torch 

def evaluate(model, context, start_answer, end_answer, tokenizer, dic, max_length = 30):
    question_predicted = []
    with torch.no_grad():
        for i in range(max_length):
            indexed_tokens, segments_ids = tokenizer.processContextAnswer(context, start_answer, end_answer)
            output = model(indexed_tokens, segments_ids, question_predicted)
            target_index = torch.argmax(output).item()
            question_predicted.append(target_index)
            if target_index == 102:
                break
        question_predicted = " ".join([dic.word_by_id(w) for w in question_predicted])
    return question_predicted


def load_model(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.required_grad = False

    model.eval()
    return model

def test(model, dataset, dic, output_path):
    with open(output_path, "a") as file: 
        for data in dataset: 
            context = data[0]
            start_answer = data[2]
            end_answer = data[3]
            question = evaluate(model, context, start_answer, end_answer, dic)
            file.write(question)

            
if __name__ == '__main__':

    #load model from checkpoint 
    tmodel = load_model('/checkpoint/Model.pth')

    #output path
    output_path = '/output_question.txt'

    #load train dataset
    dataset = Dataset('/data/squad2.0/dev-v2.0.json')

    #load dictionnary
    dic = Dictionary('/data/tag.json', '/pre_trained/bert_base_uncased_tokenizer/vocab.txt')

    #test
    test(model, dataset, dic, output_path)
