from model import *
import json
from collections import namedtuple
from tqdm import tqdm


def generate(model, tokenizer, context, answer):
    input_ids, attention_mask = tokenizer(context=context, answer=answer)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    with torch.no_grad():
        predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=50, early_stopping=True)
        predictions = [predictions[0].tolist()]
        predictions = tokenizer.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        return predictions

if __name__ == '__main__':
    args = open('config.json').read()
    args = json.loads(args)
    args = namedtuple('args', args.keys())(*args.keys())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args)
    model = LitQGModel.load_from_checkpoint(args.checkpoint, args=args).eval().to(device)

    data = pd.read_csv(args.val_dataset)

    outputs = []
    for row in tqdm(list(data.iterrows())):
        context, answer = row[1]['context'], row[1]['text']
        predictions = generate(model.generator, tokenizer, context, answer)
        outputs.append([row[1]['question'], predictions[0] if type(predictions) is list else predictions])
    
    outputs = pd.DataFrame(data=outputs, columns=['ground_truth', 'prediction'])
    outputs.to_csv('predictions.csv', index=False)