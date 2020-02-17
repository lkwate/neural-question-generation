from utils import *
from tqdm import tqdm
from torch import optim
import torch.nn as nn
import torch
from model import TransformerModel

#Training step
def trainStep(model, optimizer, criterion, tokenizer, batch):
    optimizer.zero_grad()
    loss = 0
    for batch_item in batch:
        context = batch_item[0]
        question = batch_item[1]
        start_answer = batch_item[2]
        end_answer = batch_item[3]

        question_predicted = []
        indexed_tokens_question, segments_ids_question = tokenizer.processQuestion(question)
        indexed_tokens_context, segments_ids_context = tokenizer.processContextAnswer(context, start_answer, end_answer)

        memory, output = model(indexed_tokens_context=indexed_tokens_context, segments_ids_context=segments_ids_context,
                               indexed_tokens_question=[indexed_tokens_question[0]],
                               segments_ids_question=[segments_ids_question[0]])

        for qi in range(1, len(indexed_tokens_question) - 1):
            target_index = torch.argmax(output[-1, :]).item()
            question_predicted.append(target_index)
            loss += criterion(output[-1, :].unsqueeze(0), torch.LongTensor([indexed_tokens_question[qi]]).to(device))
            output = model(memory=memory, indexed_tokens=indexed_tokens_question[: qi + 1],
                           segments_ids=segments_ids_question[: qi + 1], decode=True)
        # last tokens
        loss += criterion(output[-1, :].unsqueeze(0), torch.LongTensor([indexed_tokens_question[-1]]).to(device))
        target_index = torch.argmax(output[-1, :]).item()
        question_predicted.append(target_index)

    # compute gradient of loss function
    loss.backward()

    # update parameter
    optimizer.step()

    del indexed_tokens_context, indexed_tokens_question, output

    return loss, question_predicted, question


def trainTBertIter(dataset, model, optimizer, criterion, tokenizer, epoch=1, iteration=1, num_epoch=10,
                   period_display=17):
    """
        loop over dataset and train on each sample.
    """
    iter = 1
    plot_losses = []
    plot_losses_epoch = []
    plot_total_loss = 0
    plot_total_loss_epoch = 0

    # number of iteration in one batch
    batch_size = constant['batch_size']
    niter = len(dataset) // batch_size
    remain = len(dataset) == niter * batch_size
    print(epoch, iteration)
    for ep in tqdm(range(epoch, constant['epoch'])):
        for j in tqdm(range(iteration, niter)):
            batch = dataset[j * batch_size: (j + 1) * batch_size]
            loss, last_predicted, last_truth = trainStep(model, optimizer, criterion, tokenizer, batch)
            plot_total_loss += loss.item()
            plot_total_loss_epoch += loss.item()

            if j % period_display == 1:
                avg_loss = plot_total_loss / (period_display * batch_size)
                plot_losses.append(avg_loss)
                last_predicted = tokenizer.decode(last_predicted)
                print("loss = %.7f " % (avg_loss))
                print("predicted : ", last_predicted)
                print("truth : ", last_truth)
                plot_total_loss = 0
        if remain:
            batch = dataset[niter * batch_size:]
            loss, last_predicted, last_truth = trainStep(model, optimizer, criterion, tokenizer, batch)
            plot_total_loss = loss.item()
            plot_total_loss_epoch += loss.item()
            avg_loss = plot_total_loss / (len(dataset) - niter * batch_size)
            plot_losses.append(avg_loss)
            plot_total_loss = 0

        plot_losses_epoch.append(plot_total_loss_epoch)
        plot_total_loss_epoch = 0

        # save model
        pathCheckpoint = './checkpointModel.pth'
        checkPointModel = {
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': ep,
            'plot_losses': plot_losses,
            'plot_losses_epoch': plot_losses_epoch
        }
        torch.save(checkPointModel, pathCheckpoint)
    return plot_losses


if __name__ == '__main__':

    #load dic
    model = TransformerModel('./pre_trained/bert_base_uncased_model/')

    #optimizer
    adamOptimizer = optim.Adam(model.parameters(), lr=constant['learning_rate'])

    #loss function
    criterion = nn.CrossEntropyLoss()

    #tokenizer
    tokenizer = Tokenizer('./pre_trained/bert_base_uncased_tokenizer')
    
    # load dataset 
    dataset = Dataset('./data/squad2.0/train-v2.0.json')


    #begin training 
    trainTBertIter(dataset.dataset, model, adamOptimizer, criterion, tokenizer, epoch = 1, iteration = 1)

