from utils import *
import torch 

def evaluate(model, context, start_answer, end_answer, dic):
    
    question = [constant['cls']]
    max_length_question = constant['max_question_length']
    with torch.no_grad(): 
        memory, prob_max = model(context, question, start_answer, end_answer)
        target_index = torch.argmax(prob_max).item()
        question.append(target_index)
        
        for di in range(1, max_length_question):
            prob_max = model.decode(question, memory)
            prob_max = prob_max[di, :].unsqueeze(0)
            target_index = torch.argmax(prob_max).item()
            question.append(target_index)
            
            if target_index == constant['sep']:
                break
                
    return [dic.word_by_id(item) for item in question]


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
        for _, data in dataset.items(): 
            context = data['context']
            for qas in data['qas']:
                start_answer = qas[1]
                end_answer = qas[2]
                question = evaluate(model, context, start_answer, end_answer, dic)
                question = " ".join(question)
                file.write(question)

            
if __name__ == '__main__':

    #load model from checkpoint 
    transformer = load_model('./checkpoint/checkpointTransformer.pth')

    #output path
    output_path = './output_question.txt'

    #load train dataset
    dataset = Dataset('./data/squad2.0/dev-v2.0.json')

    #load dictionnary
    dic = Dictionary('./data/tag.json', './pre_trained/bert_base_uncased_tokenizer/vocab.txt')

    #test
    test(transformer, dataset, dic, output_path)
