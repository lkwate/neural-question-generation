from utils import *
from transformer.model import *
from torch import optim
import torch.nn as nn 
from bertModel.model import BertQG
from utils import *


def trainBert(model, optimizer, context, question, start_answer, end_answer, criterion, tokenizer, dic):
    question_indices = tokenizer.processQuestion(question)
    
    optimizer.zero_grad()
    loss = 0
    
    question_predicted = []
    indexed_tokens, segments_ids = tokenizer.processContextAnswer(context, start_answer, end_answer)
    for qi in range(len(question_indices)):
        output = model(indexed_tokens, segments_ids, question_predicted)
        target_index = torch.argmax(output).item()
        question_predicted.append(target_index)
        loss += criterion(output, torch.LongTensor([question_indices[qi]]).to(device))
    
    #update parameter
    loss.backward()
    
    optimizer.step()
    return loss, question_predicted

def trainBertIter(dataset, model, optimizer, criterion, tokenizer, dic, epoch = 10, period_display = 10): 
    """
        loop over dataset and train on each sample.
    """
    iter = 1
    plot_losses = []
    plot_total_lost = 0
    for _ in range(epoch): 
        for data in dataset: 
            context = data[0]
            iter += 1
            question = data[1]
            start_answer = data[2]
            end_answer = data[3]
            response = context[start_answer : end_answer]
            
            #compute the loss function
            loss, predicted = trainBert(model, optimizer, context, question, start_answer, end_answer, criterion, tokenizer, dic)
            plot_total_lost += loss
            if iter % period_display == 0:
                avg_loss = plot_total_lost / period_display
                plot_losses.append(avg_loss)
                predicted = " ".join([dic.word_by_id(w) for w in predicted])
                print("loss = %.7f" % (avg_loss))
                print("predicted " , predicted)
                print("truth ", question)
                plot_total_lost = 0
    return plot_losses


if __name__ == '__main__': 
    #load dic
    dic = Dictionary('/pre_trained/bert_base_uncased_tokenizer/vocab.txt')
    
    #create model
    model = BertQG('/pre_trained/bert_base_uncased_model/')

    #optimizer
    adamOptimizer = optim.Adam(model.parameters(), lr=constant['learning_rate'])

    #loss function
    criterion = nn.CrossEntropyLoss()

    #tokenizer
    tokenizer = Tokenizer('/kaggle/input/tokenizer/bert_base_uncased_tokenizer/')
    
    # load dataset 
    dataset = Dataset('/data/squad2.0/train-v2.0.json')


    #begin training 
    #trainIter
    trainBertIter(dataset.dataset, model, adamOptimizer, criterion, tokenizer, dic)

    #create checkpoint
    pathTransformerCheckpoint = '/checkpoint/checkpointModel.pth'
    checkpoint = {
        'model' : model,
        'state_dict' : model.state_dict()
    }

    #save model
    torch.save(checkpoint, pathTransformerCheckpoint)
