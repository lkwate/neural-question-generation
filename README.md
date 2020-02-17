# question-generation
Our system is based on BERT model and Transformer, which generate question for question answering based on text corpus. 

Some person who have knowledge about current literature of natural language processing may be wondering that our system is based on BERT model and Transformer separately.  


## Introduction 
In our educational system, the common form of examination is question forms, the teacher is generally tasked to design question forms, for student. To reach the general education purpose which try to improve understanding of the student about a specific knowledge, it is important to design the relevant questions to evaluate the skills of the student in this specific knowledge. We designed a system which try to produce tex comprehension question based on a given corpus. the system designed is unilingual (English). the system will be benefit for educators by saving his time for generate question forms.  

## To do
* Build another sub-systems answer question
* Merge the two systems to form an auto evaluation system 

## Description
### Architecture of model
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/architecture.jpg)

## Dependencies

```
pytorch 1.4.0
python 3.7 
nltk
pytorch-pretrained-bert
pytorch-transformers
```
to use our systems you need to follow those steps : 

* install dependencies : 

	```
	sudo apt-get install python3.7
	sudo apt-get install python3-pip
	pip3 install torch torchvision (https://pytorch.org/get-started/locally/ to more personalize setup)
	pip3 install nltk 
	pip3 install pytorch-pretrained-bert
	pip3 install pytorch-transformers
	pip3 install flask-cors
	```
## Dataset
we use the recently updated SQUAD v.2

## Train
```
git clone https://github.com/lkwate/neural-question-generation.git
cd neural-question-generation
python3 train.py
```

this will take a few days depending on the architecture and power of your computer

## How to use 
```
cd neural-question-generation
FLASK_ENV=development python3 launch.py
```
go to link [welcome](http://127.0.0.1:5000)

### add context
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/add_context.png)
### select answer in context
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/select_spanned_answer.png)
### add answer
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/add_answer.png)
### generate question 
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/generate_question.png)

## Result

<h2>Some predictions on testing step :</h2>
<table>
  <tr>
    <th>Context</th>
    <th> The French administration, reluctant to return their pre-war possessions to German companies, reassigned some of them to French companies. This was particularly the case for the Société financière des caoutchoucs, which obtained plantations put into operation during the German period and became the largest company in Cameroon under French mandate. Roads were being built to link the main cities together, as well as various infrastructure such as bridges and airports</th>
    <th>answer</th>
  </tr>
  <tr>
    <td rowspan="2">Question predicted</td>
    <td>what were the main types of buildings in the area?</td>
    <td>bridges and airports</td>
  </tr>
  <tr>
    <td>what were the main types of buildings in the area?</td>
    <td>Roads</td>
  </tr>
</table>

<table>
  <tr>
    <th>Context</th>
    <th>The Douala-Yaoundé railway line, begun under the German regime, had been completed. Thousands of workers were forcibly deported to this site to work fifty-four hours a week. Workers also suffered from lack of food and the massive presence of mosquitoes. In 1925, the mortality rate on the site was 61.7%. However, the other sites were not as deadly, although working conditions were generally very harsh</th>
    <th>answer</th>
  </tr>
  <tr>
    <td rowspan="4">Question predicted</td>
    <td>how were the people in the city killed?</td>
    <td>fifty-four hours a week</td>
  </tr>
  <tr>
    <td>how many people were killed in the death of the people in the city?</td>
    <td>Thousands of workers</td>
  </tr>
  <tr>
    <td>what company built the first building in the south east?</td>
    <td>German regime</td>
  </tr>
  <tr>
    <td>what was the name of the new delhi river?</td>
    <td>The Douala-Yaoundé railway line</td>
  </tr>
</table>

<h2>The BLEU scores on dev set are :</h2>
<table>
  <tr>
    <th></th>
    <th>BLEU-1</th>
    <th>BLEU-2</th>
    <th>BLEU-3</th>
    <th>BLEU-4</th>  
  </tr>
  <tr>
    <td>Dev set</td>
    <td>24.88</td>
    <td>22.11</td>
    <td>21.68</td>
    <td>20.51</td>
  </tr>
</table>


