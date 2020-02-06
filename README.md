# neural-question-generation
system based on Bert model, which generate question for question answering based on text corpus

## Introduction 
In our educational system, the common form of examination is question forms, the teacher is generally tasked to design question forms, for student. To reach the general education purpose which try to improve understanding of the student about a specific knowledge, it is important to design the relevant questions to evaluate the skills of the student in this specific knowledge. We designed a system which try to produce tex comprehension question based on a given corpus. the system designed is unilingual (English). the system will be benefit for educators by saving his time for generate question forms.  

## To do
* Build another sub-systems answer question
* Merge the two systems to form an auto evaluation system 

## Description
### Model
	our system is a fine tuned Bert Model for question generation task

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
	  	pip install torch torchvision
	  	pip install nltk 
	  	pip install pytorch-pretrained-bert
	  	pip install pytorch-transformers
	```
	

##  Execute
```
	  	git clone https://github.com/lkwate/neural-question-generation.git
	  	cd neural-question-generation
```
* fill a **.json file** (e.g. context.json) which the follow schema
```
  	 {
    	  "data" : [
    	    {
    	    	"paragraphs" : [
    	          {
    	            "context" : "paragraph context",
    	            "qas" : [
    	              {
    	                "question" : "text", 
    	                "answers" : [
    	                  {
    	                    "text" : "response",
    	                    "answer_start" : index where answer begining
    	                  },{...}
    	                ]
    	              },{...}
    	            ]
    	          },{...}
    	        ]
    	    },{...}
    	 ]
    }
```
* modify the path of **.json** (in **eval.py at line 55**) from where data will be fetch to generate question 

* execute this command to launch

```
	  	pyton eval.py
```
## evolution at training


![alt text](https://github.com/lkwate/neural-question-generation/master/training.jpg)

