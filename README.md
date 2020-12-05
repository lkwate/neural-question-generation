# Reinforcement Learning Generator-Evaluator Architecture for Question Generation. 
[demo colab](https://docs.google.com/presentation/d/1oRBKXuJ3vhxhSTsidOgRadDGV985E00h8GCCga5sj6M/edit?usp=sharing)
  
The model is based on Text-to-Text Transfer Transformer framework **T5** provided Google AI, that consists of stacks of Transformer-Encoder layers (linked in BERT fashion) and Transformer-Decoder layers (also lined in BERT fashion). The training of the Generator T5 was drived by an Evaluator that optimized the Generator parameters on task-specific reward function (BLEU + GenSim) through Policy Gradient.


## Introduction 
In our educational system, the common form of examination is question forms, the teachers are generally charged to design question forms, for the students. To reach the general education purpose that aims to improve the cognitive skills of the students about a specific knowledge, it is important to design the relevant questions to evaluate the skills of the students in this specific context. We designed a system that tries to produce conditional question based on a given corpus. the system designed is unilingual (English). the system will be benefit for educators by saving their time for generating examination.  


## Description
### Model Architecture
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/model-rl.png)

## Dependencies

```
torch
langdetect
transformers
sentencepiece
gradio
wget
```

## Dataset
SQUAD v.2

## Evaluation
We used [NUBIA](https://wl-research.github.io/blog/2020/04/29/introducing-nubia.html) to evaluate our model, we got the following results
<table>
  <tr>
    <th>Model</th>
    <th> Logical Agreement </th>
    <th> Semantic relation </th>
    <th> Chance of contradiction </th>
  </tr>
  <tr>
    <th>T5-RL</th>
    <th> 59.5/100 </th>
    <th> 39/100 </th>
    <th> 18/100 </th>
  </tr>
</table>



## Some Results

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
