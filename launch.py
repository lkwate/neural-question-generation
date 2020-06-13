from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from evaluation import evaluate, generate
from utils import *
from model import QGModel

app = Flask(__name__)
CORS(app)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predit():
    if request.method == 'POST':
        ##process data
        data = request.get_json()
        context = data['context']
        answers = []
        if 'answers' in data:
            answers = data['answers']

        list_dict_answers = []
        for ans in answers:
            list_dict_answers.append({'start_ans' : ans['start_answer'], 'end_ans' : ans['end_answer']})

        response = generate(model, tokenizer, context, device, list_dict_answers)
        response = jsonify({'questions' : response})
        return response

if __name__ == "__main__":
    app.run(debug=True)
    model = QGModel(configT5_model, config['path_t5_question_generation'])
    
