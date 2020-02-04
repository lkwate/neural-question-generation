from utils import *
from transformer.model import *
from torch import optim
import torch.nn as nn 


def train(context, start_answer, end_answer, question_indices, model, optimizer, criterion): 
    """
        assumptions :
            - question indexed_tokens of question provided by bert_tokenizer (with special token)
    """
    #set grad of optimizer to zero
    optimizer.zero_grad()
    loss = 0
    
    target_length = len(question_indices)
    
    tgt = question_indices[:1]
    memory, prob_max = model(context, tgt, start_answer, end_answer)
    for di in range(1, target_length - 1):
        loss += criterion(prob_max, torch.LongTensor([question_indices[di]]))
        tgt = question_indices[:di + 1]
        prob_max = model.decode(tgt, memory)
        prob_max = prob_max[di, :].unsqueeze(0)
        target_index = torch.argmax(prob_max).item()
        
    ## teacher to output "[SEP]" at the end of each question
    loss += criterion(prob_max, torch.LongTensor([question_indices[target_length - 1]]))
    
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item() / target_length

def trainIter(dataset, model, optimizer, criterion, period_display = 2000, epoch = 1000): 
    """
        loop over dataset and train on each sample.
    """
    iter = 1
    plot_losses = []
    plot_total_lost = 0
    for _ in range(epoch): 
        for _, data in dataset.items(): 
            context = data['context']
            for qas in data['qas']:
                question = qas[0]
                question_indices = model.embedding.bertEmb.get_id_tokens(question)
                start_answer = qas[1]
                end_answer = qas[2]
                loss = train(context, start_answer, end_answer, question_indices, model, optimizer, criterion)
                plot_total_lost += loss
                if iter % period_display:
                    avg_loss = plot_total_lost / period_display
                    plot_losses.append(avg_loss)
                    print("loss = %.7f" % (avg_loss))
                    plot_total_lost = 0
    return plot_losses


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
