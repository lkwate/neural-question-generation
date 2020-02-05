from utils import *
from transformer.model import *
from torch import optim
import torch.nn as nn 


def train(context, response, question_text, question_indices, model, optimizer, criterion, dic): 
    """
        assumptions :
            - question indexed_tokens of question provided by bert_tokenizer (with special token)
    """
    #set grad of optimizer to zero
    optimizer.zero_grad()
    loss = 0

    target_length = len(question_indices)
    question = []
    memory, prob_max = model(context, response, question_text)
    for di in range(target_length - 1):
        
        target_index = torch.argmax(prob_max[di, :]).item()
        question.append(target_index)
        loss += criterion(prob_max[di, :].unsqueeze(0).to(device), torch.LongTensor([question_indices[di + 1]]).to(device))
    
        predicted = " ".join([dic.word_by_id(item) for item in question])
        truth =  " ".join([dic.word_by_id(item) for item in question_indices])
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item() / target_length, predicted, truth


def trainIter(dataset, model, optimizer, criterion, dic, period_display = 50, epoch = 1000): 
    """
        loop over dataset and train on each sample.
    """
    iter = 1
    plot_losses = []
    plot_total_lost = 0
    for _ in range(epoch): 
        for data in dataset: 
            iter += 1
            context = data['context'].lower()
            context_replace = remove_stop_words(context)
            context_replace, dic_context, revert_dic = ner_extraction(context_replace)
            
            for qas in data['qas']:
                question = qas[0]
                question = multireplace(question, dic_context)
                response = context[qas[1] : qas[2]]
                if response == '':
                    continue
                response = remove_stop_words(response.lower())
                response_replace = multireplace(response, dic_context)
                question_indices = model.embedding.bertEmb.get_id_tokens(question)
                
                loss, predicted, truth = train(context_replace, response_replace, question, question_indices, model, optimizer, criterion, dic)
                plot_total_lost += loss
                if iter % period_display == 0:
                    avg_loss = plot_total_lost / period_display
                    plot_losses.append(avg_loss)
                    print("loss = %.7f" % (avg_loss))
                    print("predicted " , predicted)
                    print("truth ", truth)
                    print("reponse ", response)
                    plot_total_lost = 0
    return plot_losses

def evaluate(model, context, start_answer, end_answer, dic):
    
    question = [constant['cls']]
    max_length_question = constant['max_question_length']
    with torch.no_grad(): 
        memory, prob_max = model(context, question, start_answer, end_answer)
        target_index = torch.argmax(prob_max).item()
        question.append(target_index)
        
        for di in range(1, max_length_question):
            prob_max = model.decode(question, memory)
            prob_max = prob_max[di, :]
            target_index = torch.argmax(prob_max).item()
            question.append(target_index)
            
            if target_index == constant['sep']:
                break
                
    return [dic.word_by_id(item) for item in question]


if __name__ == '__main__': 
    #create model to train 
    transformer = Transformer(d_model = constatnt['d_model_simple'], 
                              vocab_size = constant['vocab_size'],
                              path_to_bert_tokenizer = './pre_trained/bert_base_uncased_tokenizer/', 
                              path_to_bert_model = './pre_trained/bert_base_uncased_model/')
    #optimizer 
    adamOptimizer = optim.Adam(transformer.parameters(), lr=constant['learning_rate'])

    #loss function 
    criterion = nn.CrossEntropyLoss()

    #load train dataset
    dataset = Dataset('./data/squad2.0/train-v2.0.json')

    #load dictionnary
    dic = Dictionary('./data/tag.json', './pre_trained/bert_base_uncased_tokenizer/vocab.txt')
    
    #begin training 
    trainIter(dataset, transformer, adamOptimizer, criterion)

    #create checkpoint
    pathTransformerCheckpoint = '/checkpoint/checkpointTransformer.pth'
    checkpoint = {
        'model' : transformer,
        'state_dict' : transformer.state_dict()
    }

    #save model
    torch.save(checkpoint, pathTransformerCheckpoint)
