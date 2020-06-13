from utils import train_dataset, valid_dataset, config, tokenizer, configT5_model, device
import torch
from torch import optim
from torch.utils.data import DataLoader
from evaluation import evaluate
from model import QGModel
from tqdm import tqdm
import json

def trainEpoch(model, optimizer, train_data_loader, valid_data_loader, tokenizer, device):
    model.train()
    optimizer.zero_grad()
    losses = []

    for step, batch in enumerate(tqdm(train_data_loader)):
        input_ids_ctx = torch.stack(batch[0], dim=1).to(device)
        attention_mask_ctx = torch.stack(batch[1], dim=1).to(device)
        input_ids_qt = torch.stack(batch[2], dim=1).to(device)
        attention_mask_qt = torch.stack(batch[3], dim=1).to(device)

        loss, _, _, _ = model(input_ids_ctx, attention_mask_ctx, input_ids_qt=input_ids_qt, attention_mask_qt=attention_mask_qt)
        loss /= config['accumulation_step']
        loss.backward()

        if (step + 1) % config['accumulation_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
    score, predictions = evaluate(model, valid_data_loader, tokenizer, device)

    return score, predictions

# global training
def globalTraining(model, optimizer, train_dataset, valid_dataset, tokenizer, device):
    model.to(device)
    for epoch in range(config['epoch']):
        train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        losses, prediction = trainEpoch(model, optimizer, train_data_loader, valid_data_loader, tokenizer, device)

        # print information
        print("Epoch {}/{}, loss = {} ----> {}".format(epoch + 1, config['epoch'], losses[0], losses[-1]))
        model.t5_model.save_pretrained(config['path_t5_question_generation'])

        with open("./prediction/losses{}.json".format(epoch + 1), "w") as file:
            json.dump(losses, file)
        with open("./prediction/prediction{}.json".format(epoch + 1), "w") as file:
            json.dump(prediction, file)

if __name__ == '__main__':
    model = QGModel(configT5_model)
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, config['schedule_rate'])
    globalTraining(model, optimizer, train_dataset, valid_dataset, tokenizer, device)