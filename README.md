# question-generation
We built a system that generates the question according to a given context and the answers of the generated question lie into the context.

Our model is based on Text-to-Text Transfer Transformer framework **T5** provided Google AI, that consists of stacks of Transformer-Encoder layers (linked in BERT fashion) and Transformer-Decoder layers (also lined in BERT fashion) . We fine tuned the model by adding a new Text-to-Text task (question generation) to the model


## Introduction 
In our educational system, the common form of examination is question forms, the teachers are generally charged to design question forms, for the students. To reach the general education purpose that aims to improve the cognitive skills of the students about a specific knowledge, it is important to design the relevant questions to evaluate the skills of the students in this specific context. We designed a system that tries to produce conditional question based on a given corpus. the system designed is unilingual (English). the system will be benefit for educators by saving their time for generating examination.  

## To do
* Build another sub-systems answer question
* Merge the two systems to form an autonomous evaluation system 

## Description
### Architecture of model
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/architecture-nn.jpg)

## Dependencies

```
torch
torchvision
langdetect
python
flask-cors
tqdm
transformers
```
to use our systems you need to follow those steps : 

* install dependencies : 

	```
	sudo apt-get install python3.7
	sudo apt-get install python3-pip
	pip3 install torch torchvision (https://pytorch.org/get-started/locally/ to personalize installation)
	pip3 install transformers
	pip3 install langdetect
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

this will take around 4 hours on GPU

## How to use 
```
cd neural-question-generation
FLASK_ENV=development python3 launch.py
```
go to link [welcome](http://127.0.0.1:5000)

### add context
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/add_context.png)
### generate question 
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/generate_question.png)

## Result

<h2>Some predictions on testing step :</h2>
<table>
  <tr>
    <th>Context</th>
    <th> When Hitler launched World War II by invading Poland, British and French officials in Cameroon seized the German-owned plantations (1939-40). After the fall of France (June 1940), the country's colonial dependencies proclaimed loyalty to Marshal Petain' new Vichy regime. Central Africa was an exception</th>
    <th>answer</th>
  </tr>
  <tr>
    <td rowspan="6">Question predicted</td>
    <td>Who launched World War II by invading Poland?</td>
    <td>Hitler</td>
  </tr>
  <tr>
    <td>What war did Hitler begin by invading Poland?</td>
    <td>World War II</td>
  </tr>
  <tr>
    <td>What country did Hitler invade in World War II?</td>
    <td>Poland</td>
  </tr>
  <tr>
    <td>Who did Cameroon's colonial dependencies proclaim loyalty to?</td>
    <td>Marshal Petain</td>
  </tr>
  <tr>
    <td>What region was an exception to the Vichy regime?</td>
    <td>Central Africa</td>
  </tr>
  <tr>
    <td>What regime did the colonial dependencies of Cameroon resign to?</td>
    <td>Vichy</td>
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
    <td>What was the mortality rate in 1925?</td>
    <td>61.7%</td>
  </tr>
  <tr>
    <td>Which regime began the Douala-Yaoundé railway?</td>
    <td>German</td>
  </tr>
  <tr>
    <td>Which railroad line in the German era was completed in 1925?</td>
    <td>Douala</td>
  </tr>
  <tr>
    <td>How many workers were deported to this railroad site?</td>
    <td>Thousands</td>
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


