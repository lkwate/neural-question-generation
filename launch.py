from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from nltk.tokenize import sent_tokenize # to tokenize paragraph in sentence
from evaluation import evaluate
from utils import *
from evaluation import load_model
from model import *

app = Flask(__name__)
CORS(app)

tokenizer = Tokenizer('./pre_trained/bert_base_uncased_tokenizer')
model = load_model('./checkpoint/newCheckpoint.pth')

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predit():
    if request.method == 'POST':
        ##process data
        data = request.get_json()
        context = data['context']
        answers = data['answers']
        is_sentence_level = data['is_sentence_level']
        data_query = []
        for ans in answers:
            start_answer = ans['start_answer']
            end_answer = ans['end_answer']
            answer = ans['answer']
            index = 0
            if is_sentence_level:
                for sent in sent_tokenize(context):
                    j = sent.find(answer)
                    if j != -1 and index <= start_answer and (index + len(sent)) >= end_answer:
                        data_query.append((sent, start_answer, end_answer, answer))
                        break
                    index += (len(sent) + 1)
            else:
                data_query.append((start_answer, end_answer, answer))

        result = []
        if is_sentence_level:
            for data_item in data_query:
                question = evaluate(model, data_item[0], data_item[1], data_item[2], tokenizer)
                result.append({"question": question, "answer": data_item[3]})
        else :
            for data_item in data_query:
                question = evaluate(model, context, data_item[0], data_item[1], tokenizer)
                result.append({"question": question, "answer": data_item[2]})
        response = jsonify({'questions' : result})
        return response

if __name__ == "__main__":
    app.run(debug=True)

    
