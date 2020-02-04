# neural-question-generation
system based on Transformers, which generate question for question answering based on text corpus

## Introduction 
In our educational system, the common form of examination is question forms, the teacher is generally tasked to design question forms, for student. To reach the general education purpose which try to improve understanding of the student about a specific knowledge, it is important to design the relevant questions to evaluate the skills of the student in this specific knowledge. We designed a system which try to produce tex comprehension question based on a given corpus. the system designed is unilingual (English). 

## Description
### Model
	our system is enterily based on transformer

## Dependencies

	  	pytorch 1.4.0
	  	pytorch 3.7 
	  	spacy 
	  	en_core_web_em

to use our systems you need to follow those steps : 

* install dependencies : 

	  	sudo apt-get install python3.7
	  	sudo apt-get install python3-pip
	  	pip install -U spacy
		python -m spacy download en_core_web_sm
		pip install torch torchvision
